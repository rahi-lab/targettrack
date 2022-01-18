from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import *
import numpy as np


class CollapsibleBox(QtWidgets.QWidget):
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)

        self.toggle_button = QtWidgets.QToolButton(
            text=title, checkable=True, checked=False
        )
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonTextBesideIcon
        )
        self.toggle_button.setArrowType(QtCore.Qt.RightArrow)
        self.toggle_button.pressed.connect(self.on_pressed)

        self.toggle_animation = QtCore.QParallelAnimationGroup(self)

        self.content_area = QtWidgets.QScrollArea(
            maximumHeight=0, minimumHeight=0
        )
        self.content_area.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self.content_area.setFrameShape(QtWidgets.QFrame.NoFrame)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)

        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self, b"minimumHeight")
        )
        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self, b"maximumHeight")
        )
        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self.content_area, b"maximumHeight")
        )

    @QtCore.pyqtSlot()
    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(
            QtCore.Qt.DownArrow if not checked else QtCore.Qt.RightArrow
        )
        self.toggle_animation.setDirection(
            QtCore.QAbstractAnimation.Forward
            if not checked
            else QtCore.QAbstractAnimation.Backward
        )
        self.toggle_animation.start()

    def setContentLayout(self, layout):
        lay = self.content_area.layout()
        del lay
        self.content_area.setLayout(layout)
        collapsed_height = (
            self.sizeHint().height() - self.content_area.maximumHeight()
        )
        content_height = layout.sizeHint().height()
        for i in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(i)
            animation.setDuration(500)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)

        content_animation = self.toggle_animation.animationAt(
            self.toggle_animation.animationCount() - 1
        )
        content_animation.setDuration(500)
        content_animation.setStartValue(0)
        content_animation.setEndValue(content_height)


class Dialog(QDialog):
    def __init__(self, rundata,parent=None):
        super().__init__()
        self.buttonOk = QtGui.QPushButton('Ok',self)
        self.buttonOk.clicked.connect(self.accept)
        self.parent=parent
        layout = QtGui.QGridLayout(self)
        row=0
        if len(rundata)==0:
            layout.addWidget(QLabel("No Neural Networks Running"),row,0)
            row+=1
        else:
            for key,val in rundata.items():
                col=0
                layout.addWidget(QLabel(key), row, col,2,1)
                col+=1
                for el in val[:-1]:
                    proc,percent=el
                    percent=float(percent)
                    layout.addWidget(QLabel(proc), row, col)
                    layout.addWidget(QLabel(str(np.round(percent*100,2))+"%"), row+1, col)
                    col+=1
                st,sts=val[-1]
                layout.addWidget(QLabel(st), row, col)
                layout.addWidget(QLabel(sts), row+1, col)
                col+=1
                if val[-1][1]=="Success":
                    pullbutton=QPushButton("Pull",self)
                    pullbutton.setStyleSheet("background-color: green")
                    pullbutton.clicked.connect(self.make_pullfunc(key,True))
                elif val[-1][1]=="Failed":
                    pullbutton=QPushButton("Delete",self)
                    pullbutton.setStyleSheet("background-color: red")
                    pullbutton.clicked.connect(self.make_pullfunc(key,False))
                else:
                    pullbutton=QPushButton("Waiting...",self)
                    pullbutton.setEnabled(False)

                layout.addWidget(pullbutton, row, col,2,1)
                row+=2
        layout.addWidget(self.buttonOk, row, 0)

    def make_pullfunc(self,key,success):
        def pullfunc():
            res,msg=self.parent.tracking.pull_NN_res(key,success)
            if res:
                dial=QMessageBox()
                dial.setText(msg)
                dial.exec_()
                self.close()
            else:
                errdial=QErrorMessage()
                errdial.showMessage('Pull Failed:\n'+msg)
                errdial.exec_()
        return pullfunc


class DataTypeChoice:

    @staticmethod
    def choose_data():
        msgBox = QtGui.QMessageBox()
        msgBox.setText('Is the annotation in the form of points or regions/masks??')
        msgBox.addButton(QtGui.QPushButton('Points'), 0)
        msgBox.addButton(QtGui.QPushButton('Masks'), 0)
        ret = msgBox.exec_()
        if ret == 0:
            return "points"
        if ret == 1:
            return "masks"
