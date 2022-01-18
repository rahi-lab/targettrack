# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:38:58 2019
"""

from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton, QShortcut, QComboBox, QDialog, QDialogButtonBox, QInputDialog, QLineEdit, QFormLayout, QDesktopWidget
from PyQt5 import QtGui
#from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtCore import pyqtSignal, QObject, Qt
#import PyQt package, allows for GUI interactions


# TODO AD: could go into Qthelpers or other existing doc?
class CustomDialog(QDialog):

    def __init__(self, *args, **kwargs):
        super(CustomDialog, self).__init__(*args, **kwargs)

        self.setWindowTitle("Enter ID number for new neuron")
        self.setGeometry(100,100, 500,200)
        self.center()

        self.entry1 = QLineEdit()
        self.entry1.setValidator(QtGui.QIntValidator())
        self.entry1.setMaxLength(4)
        self.entry1.setAlignment(Qt.AlignRight)

#        self.entry2 = QLineEdit()
#        self.entry2.setValidator(QtGui.QIntValidator())
#        self.entry2.setMaxLength(4)
#        self.entry2.setAlignment(Qt.AlignRight)

        flo = QFormLayout()
        flo.addRow('ID number (integer):', self.entry1)

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

#        self.layout = QVBoxLayout()
#        self.layout.addWidget(self.buttonBox
        flo.addWidget(self.buttonBox)
        self.setLayout(flo)


    def center(self):
        # geometry of the main window
        qr = self.frameGeometry()

        # center point of screen
        cp = QDesktopWidget().availableGeometry().center()

        # move rectangle's center point to screen's center point
        qr.moveCenter(cp)

        # top left of rectangle becomes top left of window centering it
        self.move(qr.topLeft())


#MB added: to get the subset of cells
class CustomDialogSubCell(QDialog):

    def __init__(self, *args, **kwargs):
        super(CustomDialogSubCell, self).__init__(*args, **kwargs)

        self.setWindowTitle("Do you want to change only this subset?")
        self.setGeometry(100,100, 500,100)
        self.center()

        self.entry1 = QLineEdit()
        self.entry1.setValidator(QtGui.QIntValidator())
        self.entry1.setMaxLength(4)
        self.entry1.setAlignment(Qt.AlignRight)

#        self.entry2 = QLineEdit()
#        self.entry2.setValidator(QtGui.QIntValidator())
#        self.entry2.setMaxLength(4)
#        self.entry2.setAlignment(Qt.AlignRight)

        flo = QFormLayout()


        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

#        self.layout = QVBoxLayout()
#        self.layout.addWidget(self.buttonBox
        flo.addWidget(self.buttonBox)
        self.setLayout(flo)

    def center(self):
        # geometry of the main window
        qr = self.frameGeometry()

        # center point of screen
        cp = QDesktopWidget().availableGeometry().center()

        # move rectangle's center point to screen's center point
        qr.moveCenter(cp)

        # top left of rectangle becomes top left of window centering it
        self.move(qr.topLeft())
