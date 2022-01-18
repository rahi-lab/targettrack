#still a wrapper
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from . import gui
from .datasets_code.DataSet import DataSet
from . import main_controller


#This parses the settings as a dictionary
def loaddict(fn):
    set_dict={}
    with open(fn,"r") as f:
        for line in f:
            if line=="" or line=="\n":
                continue
            a=line.strip().split("=")
            assert len(a)==2, "Error loading"+fn
            set_dict[a[0]]=a[1]
    return set_dict


#this is the main class handling the window
class gui_single(QWidget):
    def __init__(self,dset_path):
        super().__init__()
        #settings take care of the GUI settings.
        self.settings=loaddict(os.path.join("src","current_settings.dat"))
        #this parses the properties of dataset. as above, we want to handle this with a h5 file later
        self.dataset = DataSet.load_dataset(dset_path)
        self.dataset.dset_path_from_GUI = dset_path   #MB added for solving the bug when using NN ("run_NNmasks" in Controller)
        # create the main controller
        self.controller = main_controller.Controller(self.dataset, self.settings)
        #now we call the UI init
        self.initUI()
        self.controller.set_up()

    def initUI(self):
        #this sizes the main window proportionally to the screen
        self.centerandresize()
        self.setWindowTitle('Simple Annotation GUI - '+self.dataset.name)
        self.setWindowIcon(QIcon(os.path.join("src","icon.png")))
        #this is the main widget
        self.gui = gui.gui(self.settings, self.controller)   # create the tracking utility
        Layout = QHBoxLayout()
        Layout.addWidget(self.gui)
        self.setLayout(Layout)
        self.show()

    def centerandresize(self,rat=[0.7,0.8]):
        geo=QDesktopWidget().availableGeometry()
        w,h=geo.width(),geo.height()
        self.resize(w*rat[0],h*rat[1])
        qr = self.frameGeometry()
        cp = geo.center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    #this handles the closing button event.
    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, "Closing",
            "Save remaining annotations and Neural Networks",
            QMessageBox.Save | QMessageBox.Close | QMessageBox.Cancel,
            QMessageBox.Save)
        if reply == QMessageBox.Close:
            ok,msg=self.gui.close("force","")
            if ok:
                self.dataset.close()
                print()
                print("Closing:\n"+msg)
                event.accept()
            else:
                errdial=QErrorMessage()
                errdial.showMessage('Bug: cfpark00@gmail.com')
                errdial.exec_()
        elif reply ==QMessageBox.Save:
            ok,msg=self.gui.close("save","")
            if ok:
                self.dataset.close()
                print()
                print("Closing:\n"+msg)
                event.accept()
            else:
                errdial=QErrorMessage()
                errdial.showMessage('Save failed:\n'+msg)
                errdial.exec_()
                event.ignore()
        elif reply ==QMessageBox.Cancel:
            event.ignore()
        else:
            assert False,"bug cfpark00@gmail.com"
