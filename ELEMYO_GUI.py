# Graphical interface for signal visualization and interaction with ELEMYO sensors
# 2020-06-15 by ELEMYO (https://github.com/ELEMYO/ELEMYO GUI)
# 
# Changelog:
#     2020-06-15 - initial release

# Code is placed under the MIT license
# Copyright (c) 2020 ELEMYO
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ===============================================

from PyQt5 import QtCore, QtWidgets, QtGui
import sys
import serial
import pyqtgraph as pg
import numpy as np
import time
from scipy.signal import butter, lfilter
from scipy import signal
from scipy.fftpack import fft
import serial.tools.list_ports

# Main window
class GUI(QtWidgets.QMainWindow):
    # Initialize constructor
    def __init__(self):
          super(GUI, self).__init__()
          self.initUI()
    # Custom constructor
    def initUI(self): 
        # Values
        COM='' #Example: COM='COM6'
        baudRate = 115200 #Serial frequency
        self.cycleNumber = 0; #Number of data reading cycle
        self.l = 1 #Current point
        self.dt = 0 #Updating time ms
        self.fs = 0 #Updating frequency in Hz
        self.passLowFrec = 10.0 #Low frequency for passband filter
        self.passHighFrec = 100.0 #Low frequency for passband filter
        self.Time = np.array([0]) #Tine array
        self.Data = np.array([0]) #Data array
        self.DataSP = np.array([0]) #Data for spectrum analyzer
        self.DataSP_l = 1 #length of DataSP
        self.xScaleOld = 1; #Needs for X rescale for spectrum plot
        self.dataWidth = 30000 #Maximum count of data points
        self.timeWidth = 10 #Time width of plot
        self.FFT = 0
        self.msg_end = 0
        self.imgTranslateNum = 0 #Value of spectrogram translation
        self.setWindowTitle("EMG/ECG/EEG - Analyser v1.0.1 | ELEMYO" + "    ( COM Port not found )")
        self.setWindowIcon(QtGui.QIcon('img/icon.png'))
        # Menu panel
        startAction = QtWidgets.QAction(QtGui.QIcon('img/start.png'), 'Start (Enter)', self)
        startAction.setShortcut('Return')
        startAction.triggered.connect(self.start)
        stopAction = QtWidgets.QAction(QtGui.QIcon('img/pause.png'), 'Stop (Space)', self)
        stopAction.setShortcut('Space')
        stopAction.triggered.connect(self.stop)
        refreshAction = QtWidgets.QAction(QtGui.QIcon('img/refresh.png'), 'Refresh (R)', self)
        refreshAction.setShortcut('r')
        refreshAction.triggered.connect(self.refresh)
        exitAction = QtWidgets.QAction(QtGui.QIcon('img/out.png'), 'Exit (Esc)', self)
        exitAction.setShortcut('Esc')
        exitAction.triggered.connect(self.close)
        # Toolbar
        toolbar = self.addToolBar('Tool')
        toolbar.addAction(startAction)
        toolbar.addAction(stopAction)
        toolbar.addAction(refreshAction)
        toolbar.addAction(exitAction)
        # Plot widget for main graphic
        self.pw1 = pg.PlotWidget(background = (13 , 13, 13, 255))
        self.pw1.showGrid(x = True, y = True, alpha = 0.7) 
        self.pw1.setLabel('bottom', 'Time', 's')
        self.pw1.setLabel('left', 'Data')
        self.p1 = self.pw1.plot()
        self.p1.setPen(color=(100,255,255), width=1)
        # Plot widget for FFT graphic
        self.pw2 = pg.PlotWidget(background = (13 , 13, 13, 255))
        self.pw2.showGrid(x = True, y = True, alpha = 0.7) 
        self.pw2.setLabel('bottom', 'Frequency', 'Hz')
        self.pw2.setLabel('left', 'Magnitude')
        self.p2 = self.pw2.plot()
        self.p2.setPen(color=(100,255,255), width=1)
        # Plot widget for spectrogram
        self.pw3 = pg.GraphicsLayoutWidget()
        self.pw3.setBackground(background = (13 , 13, 13, 255))
        self.p3 = self.pw3.addPlot()
        self.img = pg.ImageItem()
        self.p3.addItem(self.img)
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.pw3.addItem(self.hist)
        self.hist.gradient.restoreState(
                {'mode': 'rgb',
                 'ticks': [(0.0, (15 , 15, 15, 255)),
                           (0.01, (0, 43, 0)),
                           (0.5, (68, 255, 0)),
                           (1.0, (255, 255, 0))]})
        self.p3.setLabel('bottom', "Time", units='s')
        self.p3.setLabel('left', "Frequency", units='Hz')
        self.p3.setYRange(0, 200)
        self.p3.setXRange(0, 10)        
        # Styles
        centralStyle = "color: rgb(255, 255, 255); background-color: rgb(13, 13, 13);"
        editStyle = "border-style: solid; border-width: 1px;"
        # Settings zone
        gainText = QtWidgets.QLabel("GAIN")
        filtersText = QtWidgets.QLabel("FILTERS")
        passLowFrecText = QtWidgets.QLabel("LOW FREQUENCY: ")
        passHighFrecText = QtWidgets.QLabel("HIGH FREQUENCY: ")
        self.passLowFreq = QtWidgets.QLineEdit('10', self)
        self.passLowFreq.setMaximumWidth(100)
        self.passLowFreq.setStyleSheet(editStyle)
        self.passHighFreq = QtWidgets.QLineEdit('100', self)
        self.passHighFreq.setMaximumWidth(100)
        self.passHighFreq.setStyleSheet(editStyle)
        self.bandpass = QtWidgets.QCheckBox("BANDPASS FILTER:")
        self.bandstop50 = QtWidgets.QCheckBox("NOTCH 50 Hz")
        self.bandstop60 = QtWidgets.QCheckBox("NOTCH 60 Hz")
        gainx1 = QtWidgets.QRadioButton('x1')
        gainx1.setChecked(True)
        gainx1.Value = 1
        gainx2 = QtWidgets.QRadioButton('x2')
        gainx2.Value = 2
        gainx4 = QtWidgets.QRadioButton('x4')
        gainx4.Value = 4
        gainx5 = QtWidgets.QRadioButton('x5')
        gainx5.Value = 5
        gainx8 = QtWidgets.QRadioButton('x8')
        gainx8.Value = 8
        gainx10 = QtWidgets.QRadioButton('x10')
        gainx10.Value = 10
        gainx16 = QtWidgets.QRadioButton('x16')
        gainx16.Value = 16
        gainx32 = QtWidgets.QRadioButton('x32')
        gainx32.Value = 32
        self.button_group = QtWidgets.QButtonGroup()
        self.button_group.addButton(gainx1)
        self.button_group.addButton(gainx2)
        self.button_group.addButton(gainx4)
        self.button_group.addButton(gainx5)
        self.button_group.addButton(gainx8)
        self.button_group.addButton(gainx10)
        self.button_group.addButton(gainx16)
        self.button_group.addButton(gainx32)
        self.button_group.buttonClicked.connect(self._on_radio_button_clicked)
        # Main widget
        centralWidget = QtWidgets.QWidget()
        centralWidget.setStyleSheet(centralStyle)
        # Layout
        hbox = QtWidgets.QHBoxLayout()
        vbox = QtWidgets.QVBoxLayout()
        hbox.addWidget(self.pw2,)
        hbox.addWidget(self.pw3,)
        vbox.addWidget(self.pw1,)
        vbox.addLayout(hbox)

        gainLayout = QtWidgets.QGridLayout()
        gainLayout.addWidget(gainx1,0,0)
        gainLayout.addWidget(gainx2,0,1)
        gainLayout.addWidget(gainx4,0,2)
        gainLayout.addWidget(gainx5,1,0)
        gainLayout.addWidget(gainx8,1,1)
        gainLayout.addWidget(gainx10,1,2)
        gainLayout.addWidget(gainx16,2,0)
        gainLayout.addWidget(gainx32,2,1) 
        layout = QtWidgets.QGridLayout()
        layout.addWidget(gainText,0,0) 
        layout.addWidget(filtersText,0,1) 
        layout.addLayout(gainLayout,1,0,2,1)
        layout.addWidget(self.bandpass,1,2) 
        layout.addWidget(passLowFrecText,2,2)
        layout.addWidget(self.passLowFreq,2,3) 
        layout.addWidget(passHighFrecText,2,4) 
        layout.addWidget(self.passHighFreq,2,5)  
        layout.addWidget(self.bandstop50,1,1) 
        layout.addWidget(self.bandstop60,2,1) 
        
        vbox.addLayout(layout)
        centralWidget.setLayout(vbox)
        self.setCentralWidget(centralWidget)  
        self.showMaximized()
        self.show()
        # Serial monitor
        self.monitor = SerialMonitor(COM, baudRate)
        self.monitor.bufferUpdated.connect(self.updateListening, QtCore.Qt.QueuedConnection)
    # Start working
    def start(self):
        self.monitor.running = True
        self.monitor.start()
    # Pause
    def stop(self):
        self.monitor.running = False    
    # Refresh
    def refresh(self):
        if self.cycleNumber<=20:
            self.cycleNumber = 0;
        self.l = 1 #Current point
        self.Time = np.array([0])
        self.Data = np.array([0]) #Data array
        self.DataSP = np.array([0]) #Spectrum data array
        self.DataSP_l = 1 #length for DataSP
        self.msg_end = 0        
        self.imgTranslateNum = -self.imgTranslateNum
    # Update
    def updateListening(self, msg):
        # Update variables
        if (self.cycleNumber == 0):
            self.setWindowTitle("EMG/ECG/EEG - Analyser v1.0.1 | ELEMYO " + 
                                "    ( " + self.monitor.COM + " , " + str(self.monitor.baudRate) + " baud )")
        s = self.passLowFreq.text()
        if s.isdigit():
            self.passLowFrec = float(s)
        s = self.passHighFreq.text()
        if s.isdigit():
            self.passHighFrec = float(self.passHighFreq.text())
        # Parsing data from serial buffer
        msg = msg.decode(errors='ignore')
        self.cycleNumber += 1
        if (self.cycleNumber > 10 and self.dt == 0):
            self.dt = 1270 / 1000000
            self.fs = 1 / self.dt
        if len(msg) >= 2:
            if (msg[0] != '\n' and msg[1] != '\n' and self.msg_end != '\n'):
                s = str(self.Data[self.l - 1]) + msg[0 : msg.find('\r',1)]
                if (s !=''):
                    if (s.find('T') < 0 and self.l > 0 and s.isdigit()):
                        self.Data[self.l - 1]=int(s)
                    else:
                        s = s[s.find('T') + 1 :]
                        if  s.isdigit():
                            self.dt = int(s) / 1000000
                            self.fs = 1 / self.dt
                msg = msg[msg.find('\r', 1) : len(msg)]
            self.msg_end = msg[len(msg) - 1]
            for s in msg.split('\r\n'):
                if (s != '') :
                    if (s.find('T') >= 0) :
                        s = s[s.find('T') + 1 :]
                        if  s.isdigit():
                            self.dt = int(s) / 1000000
                            self.fs = 1 / self.dt
                        continue                
                    if (self.dt > 0):
                        self.Data = np.append(self.Data, int(s))
                        self.Time = np.append(self.Time, self.Time[self.l - 1] + self.dt)
                        self.l = self.l + 1
                        self.DataSP_l = self.DataSP_l + 1
            if ( len(self.Data) > self.dataWidth) :
                index = list(range(0, len(self.Data) - self.dataWidth))
                self.Data = np.delete(self.Data, index)
                self.Time = np.delete(self.Time, index)
                self.l = self.l - len(index)
        # Filtering
        Data = self.Data
        if (self.bandpass.isChecked() == 1 and self.passLowFrec < self.passHighFrec 
            and self.passLowFrec > 0 and self.fs > 1.1*self.passHighFrec):
            Data = self.butter_bandpass_filter(Data, self.passLowFrec, self.passHighFrec, self.fs)
        if self.bandstop50.isChecked() == 1:
            if self.fs > 110:
                Data = self.butter_bandstop_filter(Data, 48, 52, self.fs)
            if self.fs > 210:
                Data = self.butter_bandstop_filter(Data, 98, 102, self.fs)
            if self.fs > 310:
                Data = self.butter_bandstop_filter(Data, 148, 152, self.fs)
            if self.fs > 410:
                Data = self.butter_bandstop_filter(Data, 198, 202, self.fs)
        if self.bandstop60.isChecked() == 1:
            if self.fs > 130:
                Data = self.butter_bandstop_filter(Data, 58, 62, self.fs)
            if self.fs > 230:
                Data = self.butter_bandstop_filter(Data, 118, 122, self.fs)
            if self.fs > 330:
                Data = self.butter_bandstop_filter(Data, 158, 162, self.fs)
            if self.fs > 430:
                Data = self.butter_bandstop_filter(Data, 218, 222, self.fs)
        # Shift the boundaries of the graph
        timeCount = self.Time[self.l - 1] // self.timeWidth
        # Update plot
        if (self.l > 3):
            # Main signal graphic
            self.pw1.setXRange(self.timeWidth * timeCount, self.timeWidth * ( timeCount + 1))
            self.p1.setData(y=Data[1: -2], x = self.Time[1: -2])
            # FFT graphic
            if len(Data) > 1000:
                Y = abs(fft(Data[-1000: -2])) / 998
                X = 1/self.dt*np.linspace(0, 1, 998)
                self.FFT = (1-0.85)*Y + 0.85*self.FFT
                self.p2.setData(y=self.FFT[2: int(len(self.FFT)/2)], x=X[2:int(len(X)/2)])   
            # Spectrogram
            if self.DataSP_l >= 256:
                self.DataSP = Data[-self.DataSP_l:]
                f, t, Sxx = signal.spectrogram(self.DataSP[0: -2], self.fs)
                self.hist.setLevels(np.min(Sxx), np.max(Sxx))
                self.p3.setXRange(self.timeWidth * timeCount, self.timeWidth * ( timeCount + 1))
                
                if self.cycleNumber == 10 :
                    self.img.scale(1, f[-1] / np.size(Sxx, axis = 0))
                self.img.scale(t[-1] / np.size(Sxx, axis = 1) / self.xScaleOld,1)
                self.xScaleOld = t[-1] / np.size(Sxx, axis = 1)     
                
                if ( len(self.DataSP) <= self.dataWidth - 50) :
                    self.img.setImage(Sxx.T)
                else:
                    self.DataSP=np.array([0])
                    self.imgTranslateNum += self.DataSP_l * self.dt
                    self.img.translate(self.DataSP_l * self.dt / t[-1] * np.size(Sxx, axis = 1), 0)
                    self.DataSP_l = 0;
                if self.imgTranslateNum < 0:
                    self.img.translate(self.imgTranslateNum/t[-1] * np.size(Sxx, axis = 1), 0)
                    self.imgTranslateNum = 0
                    
    # Values for butterworth bandpass filter
    def butter_bandpass(self, lowcut, highcut, fs, order = 4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype = 'bandpass')
        return b, a
    # Butterworth bandpass filter
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order = 4):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    # Values for butterworth bandstop filter
    def butter_bandstop(self, lowcut, highcut, fs, order = 2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype = 'bandstop')
        return b, a
    # Butterworth bandstop filter
    def butter_bandstop_filter(self, data, lowcut, highcut, fs, order = 4):
        b, a = self.butter_bandstop(lowcut, highcut, fs, order = order)
        y = lfilter(b, a, data)
        return y
    # Change gain
    def _on_radio_button_clicked(self, button):
        if self.monitor.COM != '':
            self.monitor.ser.write(bytearray([button.Value]))
    # Exit event
    def closeEvent(self, event):
        self.monitor.ser.close()
        event.accept()
        
# Serial monitor class
class SerialMonitor(QtCore.QThread):
    bufferUpdated = QtCore.pyqtSignal(bytes)
    # Custom constructor
    def __init__(self, COM, baudRate):
        QtCore.QThread.__init__(self)
        self.running = False
        self.filter = False
        self.COM = COM
        self.baudRate = baudRate
        self.checkPort = 1

    # Listening port
    def run(self):
        while self.running is True:
            while self.COM == '': 
                ports = serial.tools.list_ports.comports(include_links=False)
                for port in ports :
                    self.COM = port.device
                if self.COM != '':
                    time.sleep(0.5)
                    self.ser = serial.Serial(self.COM, 115200)
                    self.checkPort = 0
            while self.checkPort:
                ports = serial.tools.list_ports.comports(include_links=False)
                for port in ports :
                    if self.COM == port.device:
                        time.sleep(0.5)
                        self.ser = serial.Serial(self.COM, self.baudRate)
                        self.checkPort = 0
                   
            # Waiting for data
            while (self.ser.inWaiting() == 0):
                pass
            # Reading data
            msg = self.ser.read( self.ser.inWaiting() )
            if msg:
                #Parsing data
                self.bufferUpdated.emit(msg)
                time.sleep(0.05)
                
# Starting program       
if __name__ == '__main__':
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    window = GUI()
    window.show()
    window.start()
    sys.exit(app.exec_())
