# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.14.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1737, 861)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 1151, 811))
        self.label.setMouseTracking(False)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setObjectName("label")
        self.openCameraButton = QtWidgets.QPushButton(self.centralwidget)
        self.openCameraButton.setGeometry(QtCore.QRect(1180, 0, 81, 131))
        self.openCameraButton.setObjectName("openCameraButton")
        self.closeButton = QtWidgets.QPushButton(self.centralwidget)
        self.closeButton.setGeometry(QtCore.QRect(1570, 0, 81, 131))
        self.closeButton.setObjectName("closeButton")
        self.openAudioButton = QtWidgets.QPushButton(self.centralwidget)
        self.openAudioButton.setGeometry(QtCore.QRect(1270, 0, 91, 131))
        self.openAudioButton.setObjectName("openAudioButton")
        self.openVideoButton = QtWidgets.QPushButton(self.centralwidget)
        self.openVideoButton.setGeometry(QtCore.QRect(1370, 0, 91, 131))
        self.openVideoButton.setObjectName("openVideoButton")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(1180, 160, 561, 651))
        self.textBrowser.setMouseTracking(True)
        self.textBrowser.setTabletTracking(True)
        self.textBrowser.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser.setObjectName("textBrowser")
        self.recordButton = QtWidgets.QPushButton(self.centralwidget)
        self.recordButton.setGeometry(QtCore.QRect(1470, 0, 91, 131))
        self.recordButton.setObjectName("recordButton")
        self.exitButton = QtWidgets.QPushButton(self.centralwidget)
        self.exitButton.setGeometry(QtCore.QRect(1660, 0, 81, 131))
        self.exitButton.setObjectName("exitButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1737, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.openCameraButton.setText(_translate("MainWindow", "打开摄像头"))
        self.closeButton.setText(_translate("MainWindow", "关闭"))
        self.openAudioButton.setText(_translate("MainWindow", "导入音频"))
        self.openVideoButton.setText(_translate("MainWindow", "导入视频"))
        self.recordButton.setText(_translate("MainWindow", "录制"))
        self.exitButton.setText(_translate("MainWindow", "退出"))
