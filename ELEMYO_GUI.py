# Graphical interface for signal visualization and interaction with ELEMYO sensors
# 2022-06-16 by ELEMYO (https://github.com/ELEMYO/ELEMYO GUI)
# 
# Changelog:
#     2022-06-29 - improved playback
#     2022-06-16 - improved sample rate
#     2022-06-01 - improved user interface
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

import sys
import os
import pkg_resources

required = {'pyserial', 'pyqtgraph', 'pyqt5', 'numpy', 'scipy'} 
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    for module in missing:
        if module == "pyqt5": module += "==5.15.5"
        os.system("python -m pip install " + module)

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt
import serial
import pyqtgraph as pg
import numpy as np
import time
from scipy.signal import butter, lfilter
import serial.tools.list_ports
from scipy.fftpack import fft
from serial import SerialException
from datetime import datetime

# Main window
class GUI(QtWidgets.QMainWindow):
    # Initialize constructor
    def __init__(self):
          super(GUI, self).__init__()
          self.initUI()
    # Custom constructor 
    def initUI(self): 
        # Values
        self.delay = 0.07 # Graphics update delay
        self.setWindowTitle("ELEMYO GUI v1.1.1")
        self.setWindowIcon(QtGui.QIcon('img/icon.png'))
        
        self.fs = 2000 # Sampling frequency in Hz
        self.dt = 1/self.fs  # Time between two signal sempling in s  
        
        self.passLowFrec = 10 # Low frequency for band-pass filter
        self.passHighFrec = 200 # High frequency for band-pass filter
        
        self.dataWidth = int(12/self.dt) # Maximum count of plotting data points (12 seconds window)
        self.Data = np.zeros((6, self.dataWidth)) # Raw data array, first index - sensor number, second index - sensor data
        self.DataEnvelope = np.zeros((6, self.dataWidth)) # Envelope of row data, first index - sensor number, second index - sensor data
        self.l = 0 # Current sensor data point
        self.Time = [0]*self.dataWidth# Time array (in seconds)
        self.timeWidth = 10 # Plot window length in seconds
        self.sampleNum = 0
        
        self.MovingAverage = MovingAverage(self.fs) # Variable for data envelope (for moving average method)
        
        self.recordingFileName = '' # Recording file name
        self.loadFileName = '' # Data load file name
        self.loadFile = 0 # Data load variable
        self.sliderpos = 0 # Position of data slider 
        self.StartTime = 0
        self.EndTime = 0
        
        self.FFT = np.zeros((6, 2000)) # Fast Fourier transform data
        
        # Accessory variables for data read from serial
        self.ms_len = 0;
        self.msg_end = bytearray([0])
        
        # Menu panel
        self.COMports=QtWidgets.QComboBox()
        self.COMports.setDisabled(False)
        self.COMports.setSizeAdjustPolicy(0)
        
        self.liveFromSerialAction = QtWidgets.QAction(QtGui.QIcon('img/play.png'), 'Start\Stop live from serial ', self)
        self.liveFromSerialAction.setCheckable(True)
        self.liveFromSerialAction.setChecked(False)
        self.liveFromSerialAction.triggered.connect(self.liveFromSerial)
        
        self.dataRecordingAction = QtWidgets.QAction(QtGui.QIcon('img/rec.png'), 'Start/Stop recording', self)
        self.dataRecordingAction.triggered.connect(self.dataRecording)
        self.dataRecordingAction.setCheckable(True)
        self.dataRecordingAction.setDisabled(True)
        
        self.refreshAction = QtWidgets.QAction(QtGui.QIcon('img/refresh.png'), 'Refresh screen (R)', self)
        self.refreshAction.setShortcut('r')
        self.refreshAction.triggered.connect(self.refreshForAction)
        self.refreshAction.setDisabled(True)   
        
        self.pauseAction = QtWidgets.QAction(QtGui.QIcon('img/pause.png'), 'Pause (Space)', self)
        self.pauseAction.setCheckable(True)
        self.pauseAction.setChecked(False)
        self.pauseAction.setDisabled(True)
        self.pauseAction.triggered.connect(self.pause)
        self.pauseAction.setShortcut('Space')
        
        dataLoadAction = QtWidgets.QAction(QtGui.QIcon('img/load.png'), 'Select playback file', self)
        dataLoadAction.triggered.connect(self.dataLoad)
               
        self.PlaybackAction = QtWidgets.QAction(QtGui.QIcon('img/playback.png'), 'Start/Stop playback from file: \nFILE NOT SELECTED', self)
        self.PlaybackAction.triggered.connect(self.Playback)
        self.PlaybackAction.setCheckable(True)
        self.PlaybackAction.setDisabled(True)
        
        self.slider = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.slider.setValue(0)
        self.slider.setFixedWidth(40)
        self.slider.setDisabled(True) 
        
        self.sensorsNumberAction = QtWidgets.QLabel('SENSORS NUMBER: ', self)
        self.sensorsNumberAction1 = QtWidgets.QLabel('     ', self)
        self.sensorsNumber = QtWidgets.QDoubleSpinBox()
        self.sensorsNumber.setRange(1, 6)
        self.sensorsNumber.setDecimals(0)
        self.sensorsNumber.setDisabled(True)
        
        self.SignalType = QtWidgets.QLabel('       SIGNAL TYPE: ', self)
        
        self.SignalTypeBox=QtWidgets.QComboBox()
        self.SignalTypeBox.addItem("RAW ADC DATA")
        self.SignalTypeBox.addItem("INITIAL (in μV)")
        self.SignalTypeBox.setDisabled(True)
        self.SignalTypeBox.currentIndexChanged.connect(self.refresh)
        
        self.ADCType = QtWidgets.QLabel('     ADC (bits): ', self)
        
        self.ADCTypeBox=QtWidgets.QComboBox()
        self.ADCTypeBox.addItem("8")
        self.ADCTypeBox.addItem("10")
        self.ADCTypeBox.addItem("12")
        self.ADCTypeBox.addItem("14")
        self.ADCTypeBox.addItem("16")
        self.ADCTypeBox.setDisabled(True)
        self.ADCTypeBox.setCurrentIndex(1)
        
        self.rawSignalAction = QtWidgets.QCheckBox('MAIN SIGNAL', self)
        self.rawSignalAction.setChecked(True)
        self.rawSignalAction1 = QtWidgets.QLabel('       ', self)  
        
        self.EnvelopeSignalAction = QtWidgets.QCheckBox('ENVELOPE:', self)
        self.EnvelopeSignalAction.setChecked(True)
        self.EnvelopeSignalAction1 = QtWidgets.QLabel('    ', self)
        self.envelopeSmoothingСoefficient = QtWidgets.QDoubleSpinBox()
        self.envelopeSmoothingСoefficient.setSingleStep(0.01)
        self.envelopeSmoothingСoefficient.setRange(0, 1)
        self.envelopeSmoothingСoefficient.setValue(0.95)
        
        self.bandstopAction = QtWidgets.QCheckBox('BANDSTOP FILTER:', self)
        self.bandstopAction.setCheckable(True)
        
        self.notchActiontypeBox=QtWidgets.QComboBox()
        self.notchActiontypeBox.addItem("50 Hz")
        self.notchActiontypeBox.addItem("60 Hz")
        self.notchActiontypeBox.setDisabled(True)
                        
        self.bandpassAction = QtWidgets.QCheckBox('BANDPASS FILTER:', self)
        self.bandpassAction.setCheckable(True)
        self.bandpassAction.setChecked(True)
        self.bandpassAction1 = QtWidgets.QLabel('  -  ', self)
        self.bandpassAction2 = QtWidgets.QLabel('       ', self)
        
        self.passLowFreq = QtWidgets.QSpinBox()
        self.passLowFreq.setRange(10, 249)
        self.passLowFreq.setValue(10)
        self.passLowFreq.setDisabled(True)
                      
        self.passHighFreq = QtWidgets.QSpinBox()
        self.passHighFreq.setRange(10, 249)
        self.passHighFreq.setValue(200)
        self.passHighFreq.setDisabled(True)     
        
        self.sensorSelectedAction = QtWidgets.QLabel('Sensor: ', self)
        
        self.sensorSelectedActionBox=QtWidgets.QComboBox()
        self.sensorSelectedActionBox.addItem("1")

#--------------------------        
        # Toolbar
        toolbar = []
        toolbar.append(self.addToolBar('Tool1'))
        toolbar.append(self.addToolBar('Tool2'))
        toolbar.append(self.addToolBar('Tool3'))
        toolbar[0].addWidget(self.COMports)
        toolbar[0].addAction(self.liveFromSerialAction)
        toolbar[0].addAction(self.dataRecordingAction)
        toolbar[0].addAction(self.refreshAction)
        toolbar[0].addAction(self.pauseAction)
        toolbar[1].addAction(dataLoadAction)
        toolbar[1].addAction(self.PlaybackAction)
        toolbar[1].addWidget(self.slider)
        toolbar[2].addWidget(self.sensorsNumberAction)
        toolbar[2].addWidget(self.sensorsNumber)
        toolbar[2].addWidget(self.SignalType)
        toolbar[2].addWidget(self.SignalTypeBox)
        toolbar[2].addWidget(self.ADCType)
        toolbar[2].addWidget(self.ADCTypeBox)
        toolbar[2].addWidget(self.sensorsNumberAction1)
        toolbar[2].addWidget(self.rawSignalAction)
        toolbar[2].addWidget(self.EnvelopeSignalAction1)
        toolbar[2].addWidget(self.EnvelopeSignalAction)
        toolbar[2].addWidget(self.envelopeSmoothingСoefficient)
        toolbar[2].addWidget(self.rawSignalAction1)
        toolbar[2].addWidget(self.bandstopAction)
        toolbar[2].addWidget(self.notchActiontypeBox)
        toolbar[2].addWidget(self.bandpassAction2)
        toolbar[2].addWidget(self.bandpassAction)
        toolbar[2].addWidget(self.passLowFreq)
        toolbar[2].addWidget(self.bandpassAction1)
        toolbar[2].addWidget(self.passHighFreq)
        
        # Plot widgets for 1-6 sensors
        self.pw = [] # Plot widget array, index - sensor number
        self.p = [] # Raw data plot, index - sensor number
        self.pe = [] # Envelope data plot, index - sensor number
        
        for i in range(6):
            self.pw.append(pg.PlotWidget(background=(21 , 21, 21, 255)))
            self.pw[i].showGrid(x=True, y=True, alpha=0.7) 
            self.pw[i].setLabel('bottom', 'Time', 'seconds')
            self.p.append(self.pw[i].plot())
            self.pe.append(self.pw[i].plot())
            self.p[i].setPen(color=(100, 255, 255), width=0.8)
            self.pe[i].setPen(color=(255, 0, 0), width=1)

        for i in range(5):
            self.pw[i+1].setXLink(self.pw[i])
        
        # Plot widget for spectral Plot
        self.pwFFT = pg.PlotWidget(background=(13, 13, 13, 255))
        self.pwFFT.showGrid(x=True, y=True, alpha=0.7) 
        self.pFFT = self.pwFFT.plot()
        self.pFFT.setPen(color=(100, 255, 255), width=1)
        self.pwFFT.setLabel('bottom', 'Frequency', 'Hz')
        
        # Histogram widget
        self.pb = [] # Histogram item array, index - sensor number
        self.pbar = pg.PlotWidget(background=(13 , 13, 13, 255))
        self.pbar.showGrid(x=True, y=True, alpha=0.7)            
        self.pb.append(pg.BarGraphItem(x=np.linspace(1, 2, num=1), height=np.linspace(0, 2, num=1), width=0.3, pen=QtGui.QColor(153, 0, 0), brush=QtGui.QColor(153, 0, 0)))
        self.pb.append(pg.BarGraphItem(x=np.linspace(2, 3, num=1), height=np.linspace(0, 3, num=1), width=0.3, pen=QtGui.QColor(229, 104, 19), brush=QtGui.QColor(229, 104, 19)))
        self.pb.append(pg.BarGraphItem(x=np.linspace(3, 4, num=1), height=np.linspace(0, 4, num=1), width=0.3, pen=QtGui.QColor(221, 180, 10), brush=QtGui.QColor(221, 180, 10)))
        self.pb.append(pg.BarGraphItem(x=np.linspace(4, 5, num=1), height=np.linspace(0, 5, num=1), width=0.3, pen=QtGui.QColor(30, 180, 30), brush=QtGui.QColor(30, 180, 30)))
        self.pb.append(pg.BarGraphItem(x=np.linspace(5, 6, num=1), height=np.linspace(0, 6, num=1), width=0.3, pen=QtGui.QColor(11, 50, 51), brush=QtGui.QColor(11, 50, 51)))
        self.pb.append(pg.BarGraphItem(x=np.linspace(6, 7, num=1), height=np.linspace(0, 7, num=1), width=0.3, pen=QtGui.QColor(29, 160, 191), brush=QtGui.QColor(29, 160, 191)))
        self.pbar.addItem(self.pb[0])  
        self.pbar.setLabel('bottom', 'Sensor number')
        
        # Style
        centralStyle = "color: rgb(255, 255, 255); background-color: rgb(13, 13, 13);"
               
        # Buttons for selecting sensor for FFT analysis
        fftButton = []
        for i in range(6):
            fftButton.append(QtWidgets.QRadioButton(str(i + 1)))
            fftButton[i].Value = i + 1
        fftButton[0].setChecked(True)
        self.button_group = QtWidgets.QButtonGroup()
        for i in range(6):
            self.button_group.addButton(fftButton[i], i + 1)
        
        backLabel = []
        for i in range(5):
            backLabel.append(QtWidgets.QLabel(""))
            backLabel[i].setStyleSheet("font-size: 25px; background-color: rgb(21, 21, 21);")
        
        numberLabel = []
        for i in range(6):
            numberLabel.append(QtWidgets.QLabel(" " + str(i+1) + " "))
        numberLabel[0].setStyleSheet("font-size: 25px; background-color: rgb(153, 0, 0); border-radius: 14px;")
        numberLabel[1].setStyleSheet("font-size: 25px; background-color: rgb(229, 104, 19); border-radius: 14px;") 
        numberLabel[2].setStyleSheet("font-size: 25px; background-color: rgb(221, 180, 10); border-radius: 14px;")
        numberLabel[3].setStyleSheet("font-size: 25px; background-color: rgb(30, 180, 30); border-radius: 14px;")
        numberLabel[4].setStyleSheet("font-size: 25px; background-color: rgb(11, 50, 51); border-radius: 14px;")
        numberLabel[5].setStyleSheet("font-size: 25px; background-color: rgb(29, 160, 191); border-radius: 14px;")
        
        self.gainLabel  = []
        self.gainBox  = []
        for i in range(6):
            self.gainLabel.append(QtWidgets.QLabel("GAIN: 1000 x"))
            self.gainBox.append(QtWidgets.QComboBox())
        for i in range(6):
            self.gainBox[i].currentIndexChanged.connect(self.setGain)
            self.gainBox[i].addItem("1")
            self.gainBox[i].addItem("2")
            self.gainBox[i].addItem("4")
            self.gainBox[i].addItem("5")
            self.gainBox[i].addItem("8")
            self.gainBox[i].addItem("10")
            self.gainBox[i].addItem("16")
            self.gainBox[i].addItem("32")
    
       # Main widget
        centralWidget = QtWidgets.QWidget()
        centralWidget.setStyleSheet(centralStyle)
        
        self.textWindow = QtWidgets.QPlainTextEdit()
        self.textWindow.setReadOnly(True)
        
        self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "program launched\n")
        
        # Layout
        vbox = QtWidgets.QVBoxLayout()
        
        topleft = QtWidgets.QFrame()
        topleft.setFrameShape(QtWidgets.QFrame.StyledPanel)
        
        plotLayout = []
        self.row = []
        for i in range(6):
            plotLayout.append(QtWidgets.QGridLayout())
            plotLayout[i] = QtWidgets.QGridLayout()
            if i % 2 == 0: plotLayout[i].addWidget(backLabel[int(i/2)], 0, 0, 10, 1)
            plotLayout[i].addWidget(numberLabel[i], 0, 0, 10, 1, Qt.AlignVCenter)
            plotLayout[i].addWidget(self.pw[i], 0, 1, 10, 50) 
            plotLayout[i].addWidget(self.gainLabel[i], 0, 49) 
            plotLayout[i].addWidget(self.gainBox[i], 0, 50) 
            plotLayout[i].setContentsMargins(0, 0, 0, 0)
            self.row.append(QtWidgets.QWidget())
            self.row[i].setLayout(plotLayout[i])         
            self.row[i].hide()
        self.row[0].show()
        
        splitter = QtWidgets.QSplitter(Qt.Vertical)
        splitter.handle(100)
        for i in range(6): splitter.addWidget(self.row[i])
        
        layout = QtWidgets.QGridLayout()       
        layout.addWidget(splitter, 0, 0, 40, 4)
        layout.addWidget(self.pbar, 0, 4, 20, 11)
        layout.addWidget(self.pwFFT, 20, 4, 16, 11)
        layout.setColumnStretch(2, 2)
        
        layout.addWidget(self.sensorSelectedAction , 20, 13, 2, 1)
        layout.addWidget(self.sensorSelectedActionBox , 20, 14, 2, 1)
        
        layout.addWidget(self.textWindow, 36, 4, 3, 12)   
        
        vbox.addLayout(layout)
        centralWidget.setLayout(vbox)
        self.setCentralWidget(centralWidget)  
        self.showMaximized()
        self.show()    
        
        # Serial monitor
        self.serialMonitor = SerialMonitor(self.delay)
        ports = [self.COMports.itemText(i) for i in range(self.COMports.count())]
        
        for i in range(len(self.serialMonitor.ports)):
                if self.serialMonitor.ports[i] not in ports:
                    self.COMports.addItem(self.serialMonitor.ports[i])
                    
        if self.serialMonitor.COM != '':
            self.serialMonitor.serialConnect()
            self.liveFromSerialAction.setChecked(True)
            self.dataRecordingAction.setDisabled(False)
            self.sensorsNumber.setDisabled(False)
            self.SignalTypeBox.setDisabled(False)
            self.ADCTypeBox.setDisabled(False)
            self.refreshAction.setDisabled(False)
            self.pauseAction.setDisabled(False)   
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "live from " + self.serialMonitor.COM +" \n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            self.COMports.setDisabled(True)  
 
        self.sensorsNumber.valueChanged.connect(self.setSensorsNumber)       
        self.mainrun = MainRun(self.delay)
        self.mainrun.bufferUpdated.connect(self.updateListening, QtCore.Qt.QueuedConnection)  
        
    def liveFromSerial(self):
        if self.liveFromSerialAction.isChecked():
            self.refresh()
            self.serialMonitor.serialConnect()
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "live from " + self.serialMonitor.COM +" \n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            self.PlaybackAction.setChecked(False)
            self.refreshAction.setDisabled(False)   
            self.pauseAction.setDisabled(False)
            self.dataRecordingAction.setDisabled(False)
            self.dataRecordingAction.setChecked(False) 
            self.COMports.setDisabled(True)
            self.slider.setDisabled(True)
            self.slider.setFixedWidth(40)
            self.sensorsNumber.setDisabled(False)
            self.SignalTypeBox.setDisabled(False)
            self.ADCTypeBox.setDisabled(False)
            time.sleep(1) 
            self.sensorsNumber.setValue(1) 
        else:
            self.refresh()
            self.serialMonitor.serialDisconnection()
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "live stopped\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            self.refreshAction.setDisabled(True)   
            self.pauseAction.setDisabled(True)
            self.dataRecordingAction.setDisabled(True)
            self.dataRecordingAction.setChecked(False)
            self.COMports.setDisabled(False)
            self.sensorsNumber.setDisabled(True)
            self.SignalTypeBox.setDisabled(True)
            self.ADCTypeBox.setDisabled(True)
          
    # Start working
    def start(self):
        self.mainrun.running = True
        self.mainrun.start()
    
    # Pause data plotting
    def pause(self):
        if self.pauseAction.isChecked():
            self.mainrun.running = False
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "pause ON" + "\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
        else:
            self.mainrun.running = True
            self.mainrun.start()
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "pause OFF" + "\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)

    # Refresh data
    def refresh(self):
        self.l = 0
        self.sampleNum = 0
        self.Time = [0]*self.dataWidth
        self.Data = np.zeros((6, self.dataWidth))
        self.DataEnvelope = np.zeros((6, self.dataWidth))
        self.msg_end = bytearray([0])     
        self.ms_len =  0
        self.slider.setValue(0)
        self.MovingAverage = MovingAverage(self.fs)
        self.FFT = np.zeros((6, 2000))

    # Refresh screen
    def refreshForAction(self):
        self.refresh()
        self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "refresh" + "\n")
        self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)      
         
    # Initialize recording data to a file
    def dataRecording(self):
        if (self.dataRecordingAction.isChecked()):
            self.sensorsNumber.setDisabled(True)
            self.SignalTypeBox.setDisabled(True)
            self.ADCTypeBox.setDisabled(True)
            self.refreshAction.setDisabled(True)  
            self.pauseAction.setDisabled(True)  
            self.recordingFileName = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt"
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "recording to \"" + os.getcwd() +"\\" + self.recordingFileName + "\"\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            f = open(self.recordingFileName, "w") # Data file creation
            f.write(datetime.now().strftime("Date: %Y.%m.%d\rTime: %H:%M:%S") + "\r\n") # Data file name
            f.write("Sample frequency: " + str(self.fs) + "\r\n") # Data file format
            f.write("File format: \r\ntime in s | 6 sensor data points \r\n") # Data file format
            f.close()
        else:
            if not self.PlaybackAction.isChecked():
                self.refreshAction.setDisabled(False)
                self.SignalTypeBox.setDisabled(False)
                self.ADCTypeBox.setDisabled(False)
            self.pauseAction.setDisabled(False)
            self.sensorsNumber.setDisabled(False)
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "recording stopped. Result file: \"" + os.getcwd() + self.recordingFileName + "\"\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
                
    # Selecting playback file
    def dataLoad(self):
        if self.liveFromSerialAction.isChecked():
            self.sensorsNumber.setDisabled(False)
            self.dataRecordingAction.setChecked(False)
            self.refreshAction.setDisabled(False)    
            self.pauseAction.setDisabled(False)
            self.SignalTypeBox.setDisabled(False)
            self.ADCTypeBox.setDisabled(False)
        self.recordingFileName = ''
        path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open a file', '',
                                        'All Files (*.*)')
        if path != ('', ''):
            self.loadFileName = str(path[0])
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "playback file selected: " + self.loadFileName + "\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            self.PlaybackAction.setText("Start/Stop playback from file: \n" + self.loadFileName)
            self.PlaybackAction.setDisabled(False)
    
    # Playback initialization 
    def Playback(self):
        if self.PlaybackAction.isChecked():
            self.dataRecordingAction.setChecked(False)
            self.slider.setDisabled(False)
            self.slider.setFixedWidth(300)
            if self.liveFromSerialAction.isChecked():
                self.liveFromSerialAction.setChecked(False)
            self.refresh()
            self.liveFromSerialAction.setChecked(False)
            self.serialMonitor.serialDisconnection()
            self.dataRecordingAction.setDisabled(False)  
            self.refreshAction.setDisabled(True) 
            self.pauseAction.setDisabled(False)  
            self.COMports.setDisabled(False)
            self.sensorsNumber.setDisabled(False)
            self.SignalTypeBox.setDisabled(True)
            self.ADCTypeBox.setDisabled(True)
            
            self.loadFile = open(self.loadFileName, 'r')
            
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "playback from: " + self.loadFileName + "\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            
            for i in range (9):
                message = self.loadFile.readline()
                if message[0:18] == "Sample frequency: ":
                    self.fs = float(message[18:])
                    self.dataWidth = self.dataWidth*self.dt
                    self.MovingAverage.fs = self.fs
                    self.dataWidth = int(self.dataWidth*self.fs)
                    
                    self.dataWidth = int(12*self.fs)
                    self.refresh()
            message = self.loadFile.readline()
            s = message.split(' ')
            
            if len(s) == 8:
                self.StartTime = float(s[0])
                self.Time[0] = self.StartTime
                while message != '':
                    message = self.loadFile.readline()
                    s = message.split(' ')
                    if len(s) == 8: 
                        self.EndTime = float(s[0])
                    if len(s) != 8 and message != '':
                        self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "file read ERROR." + "\n")
                        self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
                        self.PlaybackAction.setChecked(False)   
            else:
                self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "file read ERROR." + "\n")
                self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
                self.PlaybackAction.setChecked(False)  
            
        else:
            self.slider.setDisabled(True)
            self.slider.setFixedWidth(40)
            self.refresh()
            self.dataRecordingAction.setDisabled(True)
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "playback stopped \n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            self.pauseAction.setDisabled(True)   
            self.sensorsNumber.setDisabled(True)
            self.SignalTypeBox.setDisabled(True)
            self.ADCTypeBox.setDisabled(True)

    # Update
    def updateListening(self):
        if (not self.liveFromSerialAction.isChecked()):
            self.serialMonitor.updatePorts()
                   
            ports = [self.COMports.itemText(i) for i in range(self.COMports.count())]
            
            for i in range(self.COMports.count()):
                if self.COMports.itemText(i) not in self.serialMonitor.ports:
                    self.COMports.removeItem(i)
                    
            for i in range(len(self.serialMonitor.ports)):
                if self.serialMonitor.ports[i] not in ports:
                    self.COMports.addItem(self.serialMonitor.ports[i])
            
            if self.serialMonitor.COM != self.COMports.currentText():
                self.serialMonitor.COM = self.COMports.currentText()
                self.serialMonitor.connect = False
        
        if len(self.COMports) == 0: 
            self.COMports.addItem("NO PORTS")
            self.COMports.setDisabled(True)
            self.liveFromSerialAction.setDisabled(True)
        else:
            self.COMports.setDisabled(False)
            self.liveFromSerialAction.setDisabled(False)
        
        if self.passLowFreq.value() > self.passHighFreq.value(): self.passLowFreq.setValue(self.passHighFreq.value())
        self.passLowFrec = self.passLowFreq.value()
        self.passHighFrec = self.passHighFreq.value()
        
        if self.bandpassAction.isChecked():
            self.passLowFreq.setDisabled(False)
            self.passHighFreq.setDisabled(False)
        else:
            self.passLowFreq.setDisabled(True)
            self.passHighFreq.setDisabled(True)
        
        if self.bandstopAction.isChecked(): 
            self.notchActiontypeBox.setDisabled(False)
        else:
            self.notchActiontypeBox.setDisabled(True)
            
        if self.EnvelopeSignalAction.isChecked():
            self.envelopeSmoothingСoefficient.setDisabled(False)
            self.MovingAverage.MA_alpha = self.envelopeSmoothingСoefficient.value()
        else:
            self.envelopeSmoothingСoefficient.setDisabled(True)
        
        # Read data from File               
        if (self.PlaybackAction.isChecked() and self.loadFileName != ''):
            self.readFromFile()
        
        # Read data from serial          
        if (self.liveFromSerialAction.isChecked()):
            self.readFromSerial()
        
        while self.sensorSelectedActionBox.count() < int(self.sensorsNumber.value()): 
            self.sensorSelectedActionBox.addItem(str(self.sensorSelectedActionBox.count() + 1))
            
        while self.sensorSelectedActionBox.count() > int(self.sensorsNumber.value()): 
            self.sensorSelectedActionBox.removeItem(self.sensorSelectedActionBox.count()-1)
        
        # Filtering
        if (self.PlaybackAction.isChecked() and self.loadFileName != '') or (self.liveFromSerialAction.isChecked()):
            Data = np.zeros((6, self.dataWidth))
            Time = np.zeros((6, self.dataWidth))
            
            for i in range( int(self.sensorsNumber.value()) ):
                Data[i] = np.concatenate((self.Data[i][self.l: self.dataWidth], self.Data[i][0: self.l]))
                Time = np.concatenate((self.Time[self.l: self.dataWidth], self.Time[0: self.l]))
            
                if self.bandstopAction.isChecked():
                    if (self.notchActiontypeBox.currentText() == "50 Hz"): 
                        for j in range(int(self.fs//100)): Data[i] = self.butter_bandstop_filter(Data[i], 45 + j*50, 55 + j*50, self.fs)
                    if (self.notchActiontypeBox.currentText() == "60 Hz"):
                        for j in range(int(self.fs//120)): Data[i] = self.butter_bandstop_filter(Data[i], 55 + j*60, 65 + j*60, self.fs)
                                
                if (self.bandpassAction.isChecked()) :
                    Data[i] = self.butter_bandpass_filter(Data[i], self.passLowFrec, self.passHighFrec, self.fs)
            
                # Shift the boundaries of the graph
                self.pw[i].setXRange(self.timeWidth*(self.Time[self.l - 1] // self.timeWidth), self.timeWidth*((self.Time[self.l - 1] // self.timeWidth + 1)))
                
                # Plot raw and envelope data
                if  self.rawSignalAction.isChecked(): self.p[i].setData(y=Data[i], x=Time)
                else: self.p[i].clear()
                
                self.DataEnvelope[i][0: self.dataWidth - self.ms_len] = self.DataEnvelope[i][self.ms_len:self.dataWidth]
                for j in range (self.dataWidth - self.ms_len, self.dataWidth):
                    self.DataEnvelope[i][j] = self.MovingAverage.movingAverage(i, Data[i][j])
                
                # Plot envelope data
                if  self.EnvelopeSignalAction.isChecked(): self.pe[i].setData(y=self.DataEnvelope[i], x=Time)
                else: self.pe[i].clear()
                    
                # Plot histogram
                self.pb[i].setOpts(height=2*self.DataEnvelope[i][-1])
            
            for i in range( int(self.sensorsNumber.value()), 6):
                self.p[i].clear()
                self.pe[i].clear()
                self.pb[i].setOpts(height=0)

            self.ms_len = 0       
            
            # Plot FFT data
            Y = np.zeros((6, 2000))
            for i in range(int(self.sensorsNumber.value())):
                Y[i] = abs(fft(Data[i][-2001: -1]))/2000
                self.FFT[i] = (1-0.85)*Y[i] + 0.85*self.FFT[i]
            X = self.fs*np.linspace(0, 1, 2000)
            sensor = self.sensorSelectedActionBox.currentIndex()
            self.pFFT.setData(y=self.FFT[sensor][2: int(len(self.FFT[sensor])/3)], x=X[2: int(len(X)/3)])
        else:
            for i in range(int(self.sensorsNumber.value())):
                self.p[i].clear()
                self.pe[i].clear()
                self.pb[i].setOpts(height=0)
            self.pFFT.clear()
    
    def setGain(self, gain):
        if self.liveFromSerialAction.isChecked():
            for i in range(int(self.sensorsNumber.value())):
                self.serialMonitor.ser.write(bytearray([(i + 1)*10 + int(self.gainBox[i].currentIndex())]))
    
    # Read data from File   
    def readFromFile(self):
        j = 0
        while j < 100:
            j += 1
        
            msg = self.loadFile.readline() 
            s = msg.split(' ')
            
            if msg == '':
                self.refresh()
                self.sliderpos = 0
                self.loadFile.seek(0, 0)
                for i in range (9):
                    self.loadFile.readline()
            else:
                if float(s[0]) >= self.EndTime:
                    self.refresh()
                    self.sliderpos = 0
                    self.loadFile.seek(0, 0)
                    for i in range (9):
                        self.loadFile.readline()
                    break
                                
                if (self.slider.value() < int((self.sliderpos - self.StartTime)/(self.EndTime - self.StartTime)*100)):
                    temp = self.slider.value()
                    self.refresh()
                    self.slider.setValue(temp)
                    self.loadFile.seek(0, 0)
                    for i in range (9):
                        self.loadFile.readline()
                    self.sliderpos = self.StartTime
                    msg = self.loadFile.readline() 
                    s = msg.split(' ')
                    
                if ((self.slider.value() > int((self.sliderpos - self.StartTime)/(self.EndTime - self.StartTime)*100))):
                    time= float(s[0])
                    while (msg != '' and self.slider.value() > int((time - self.StartTime)/(self.EndTime - self.StartTime)*100)):
                        msg = self.loadFile.readline()
                        time += self.dt
                    
                    s = msg.split(' ')
                    self.refresh()
                    self.Time[-1] = float(s[0]) - self.StartTime

                s = msg.split(' ')    
                self.sliderpos = float(s[0])
                self.slider.setValue(int((self.sliderpos - self.StartTime)/(self.EndTime - self.StartTime)*100))

                if ( self.l == self.dataWidth):
                    self.l = 0    

                for i in range(6):
                    self.Data[i][self.l] = float(s[i+1])
                self.Time[self.l] = float(s[0]) - self.StartTime
                
                if (self.dataRecordingAction.isChecked()):
                    f = open(self.recordingFileName, 'a')
                    f.write(str(round(self.Time[self.l], 6)) + " " + str(round(self.Data[0][self.l], 6)) + " " + str(round(self.Data[1][self.l], 6)) + " "
                                 + str(round(self.Data[2][self.l], 6)) + " " + str(round(self.Data[3][self.l], 6)) + " "
                                 + str(round(self.Data[4][self.l], 6)) + " " + str(round(self.Data[5][self.l], 6)) + " \n")
                    f.close() 
                
                self.l += 1
                self.ms_len += 1
                self.dt = self.Time[self.l-1] - self.Time[self.l-2]
        
    # Read data from serial                  
    def readFromSerial(self): 
        msg = self.serialMonitor.serialRead()
        
        if self.serialMonitor.connect == False:
            self.setSensorsNumber(1)
            self.sensorsNumber.setValue(1)
            self.refresh()
        
        msg_end_n = 0
        sensorsNumber = int(self.sensorsNumber.value()) + 1
        
        for i in range (len(msg)-1, 1, -1):
            if msg[i] == 0xFF and msg[i-1] == 0xFF:
                msg_end_n = i
                break
            
        if msg_end_n != len(msg):
            msg_begin = self.msg_end
            self.msg_end = msg[msg_end_n+1:len(msg)]
            msg = msg_begin + msg[0:msg_end_n]
        
        self.dt = 0
        n=0
        # Parsing data from serial buffer
        l = self.l
        if len(msg) >= 2 and (len(msg)+1)%(sensorsNumber*2+2) == 0:
            for i in range(0, len(msg)-1, sensorsNumber*2+2):
                if ( self.l == self.dataWidth):
                        self.l = 0
                for j in range(sensorsNumber - 1):
                    self.Data[j][self.l] =(int(msg[i+j*2] | msg[i+j*2+1] << 8))
            
                self.dt = (int(msg[i+(sensorsNumber-1)*2] | msg[i+(sensorsNumber-1)*2+1] << 8))/1000000     
                self.Time[self.l] = self.Time[self.l - 1] + self.dt
                self.l += 1
                self.sampleNum += 1
                self.ms_len += 1
                n += 1
            if self.sampleNum != 0 and self.Time[self.l-1] != 0 and not self.dataRecordingAction.isChecked() :
                if abs(self.fs - self.sampleNum/self.Time[self.l-1])/self.fs*100 > 5:
                    if self.sampleNum/self.Time[self.l-1] > 400:
                        self.fs = self.sampleNum/self.Time[self.l-1]
                    
                        self.dt = self.Time[self.l-1]/self.sampleNum
                        self.dataWidth = self.dataWidth*self.dt
                        self.MovingAverage.fs = self.fs
                        self.dataWidth = int(self.dataWidth/self.dt)
                        
                        self.dataWidth = int(12/self.dt)
                        self.refresh()
                else:
                    if self.sampleNum/self.Time[self.l-1] > 400:
                        self.fs = self.sampleNum/self.Time[self.l-1]

        if (self.SignalTypeBox.currentIndex() == 1 ):
            for i in range(int(self.sensorsNumber.value())):
                ADCmax = (2**(int(self.ADCTypeBox.currentText())) - 1)
                coefficient = 5000.0/ADCmax/int(self.gainBox[i].currentText())
                if l <= self.l:
                    for j in range(l, self.l): self.Data[i][j] = (self.Data[i][j] - ADCmax/2)*coefficient
                else:
                    for j in range(l, self.dataWidth): self.Data[i][j] = (self.Data[i][j] - ADCmax/2)*coefficient
                    for j in range(0, self.l): self.Data[i][j] = (self.Data[i][j] - ADCmax/2)*coefficient
        
        if (self.dataRecordingAction.isChecked()):
            f = open(self.recordingFileName, 'a')
            if l <= self.l:
                for i in range(l, self.l):
                    f.write(str(round(self.Time[i], 6)) + " " + str(round(self.Data[0][i], 6)) + " " + str(round(self.Data[1][i], 6)) + " "
                                  + str(round(self.Data[2][i], 6)) + " " + str(round(self.Data[3][i], 6)) + " "
                                  + str(round(self.Data[4][i], 6)) + " " + str(round(self.Data[5][i], 6)) + " \n")
            else:
                for i in range(l, self.dataWidth):
                    f.write(str(round(self.Time[i], 6)) + " " + str(round(self.Data[0][i], 6)) + " " + str(round(self.Data[1][i], 6)) + " "
                                  + str(round(self.Data[2][i], 6)) + " " + str(round(self.Data[3][i], 6)) + " "
                                  + str(round(self.Data[4][i], 6)) + " " + str(round(self.Data[5][i], 6)) + " \n")
                for i in range(0, self.l):
                    f.write(str(round(self.Time[i], 6)) + " " + str(round(self.Data[0][i], 6)) + " " + str(round(self.Data[1][i], 6)) + " "
                                  + str(round(self.Data[2][i], 6)) + " " + str(round(self.Data[3][i], 6)) + " "
                                  + str(round(self.Data[4][i], 6)) + " " + str(round(self.Data[5][i], 6)) + " \n")
            f.close() 
        
    # Butterworth bandpass filter
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        nyq = 0.5*fs
        low = lowcut/nyq
        high = highcut/nyq
        b, a = butter(order, [low, high], btype='bandpass')
        y = lfilter(b, a, data)
        return y
    
    # Butterworth bandstop filter
    def butter_bandstop_filter(self, data, lowcut, highcut, fs, order=4):
        nyq = 0.5*fs
        low = lowcut/nyq
        high = highcut/nyq
        b, a = butter(order, [low, high], btype='bandstop')
        y = lfilter(b, a, data)
        return y
    
    # 
    def setSensorsNumber(self, num):
        if self.liveFromSerialAction.isChecked() and self.serialMonitor.connect:
            self.serialMonitor.ser.write(bytearray([int(num)]))
            time.sleep(0.1) 
            self.serialMonitor.ser.flushInput()
        self.delay = 0.07 + 0.01*num
        self.serialMonitor.delay  = self.delay
        self.mainrun.delay = self.delay
        self.refresh()
        
        for i in range(6):
            self.row[i].hide()
            self.pw[i].getAxis('bottom').setStyle(showValues=False)
            self.pw[i].showLabel('bottom', 0)
        self.pw[int(num)-1].getAxis('bottom').setStyle(showValues=True)
        self.pw[int(num)-1].showLabel('bottom', 1)
        
        self.pbar.clear()
        for i in range(int(num)):
            self.pbar.addItem(self.pb[i])  
            self.row[i].show()
        
    # Exit event
    def closeEvent(self, event):
        self.mainrun.running = False
        self.serialMonitor.serialDisconnection()
        event.accept()

# Serial monitor class
class SerialMonitor:
    # Custom constructor
    def __init__(self, delay):
        self.running = False
        self.connect = False
        self.baudRate = 115200
        self.playFile = 0   
        self.ports = [p[0] for p in serial.tools.list_ports.comports(include_links=False) ]
        self.COM = ''
        self.ser = serial.Serial()
        if len(self.ports) > 0:
            self.COM = self.ports[0]
        
    def updatePorts(self):
        self.ports = [p[0] for p in serial.tools.list_ports.comports(include_links=False) ]
    
    def serialConnect(self):
        self.updatePorts()
        if not self.connect:
            if self.COM != '':
                try:
                    self.ser = serial.Serial(self.COM, self.baudRate)
                    self.ser.setDTR(False)
                    self.ser.setRTS(False)
                    self.connect = True             
                    time.sleep(0.1) 
                    self.ser.flushInput()
                except SerialException :
                    self.connect = False
                    
    def serialDisconnection(self):
        self.ser.close()
        self.connect = False
        
    def serialRead(self):          
        msg = bytes(0)
        try:
            msg = self.ser.read( self.ser.inWaiting() )
            self.connect = True
        except SerialException :
            self.connect = False
            try:
               self.ser.close()
               self.ser.open()
               msg = bytes(0)
            except SerialException :
                pass
            pass
        return msg

# Moving average class
class MovingAverage:
    # Custom constructor
    def __init__(self, fs):
        self.MA = np.zeros((6, 3)) 
        self.MA_alpha = 0.95
        self.Y0 = np.zeros(6)
        self.X0 = np.zeros(6)
        self.fs = fs
    
    def movingAverage(self, i, data):
        wa = 2.0*self.fs*np.tan(3.1416*1/self.fs)
        HPF = (2*self.fs*(data-self.X0[i]) - (wa-2*self.fs)*self.Y0[i])/(2*self.fs+wa)
        self.Y0[i] = HPF
        self.X0[i] = data
        data = HPF
        if data < 0:
            data = -data
        self.MA[i][0] = (1 - self.MA_alpha)*data + self.MA_alpha*self.MA[i][0];
        self.MA[i][1] = (1 - self.MA_alpha)*(self.MA[i][0]) + self.MA_alpha*self.MA[i][1];
        self.MA[i][2] = (1 - self.MA_alpha)*(self.MA[i][1]) + self.MA_alpha*self.MA[i][2];
        return self.MA[i][2]*2

# Serial monitor class
class MainRun(QtCore.QThread):
    bufferUpdated = QtCore.pyqtSignal()
    # Custom constructor
    def __init__(self, delay):
        QtCore.QThread.__init__(self)
        self.running = False
        self.playFile = 0
        self.delay = delay      

    # Listening port
    def run(self):
        while self.running is True:
            self.bufferUpdated.emit()
            time.sleep(self.delay) 
         
# Starting program       
if __name__ == '__main__':
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    window = GUI()
    window.show()
    window.start()
    sys.exit(app.exec_())
