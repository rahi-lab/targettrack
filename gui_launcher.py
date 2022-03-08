#this is the main wrapper
import sys
assert len(sys.argv)==2, "No dataset name given"#Check that the dataset name is given
from PyQt5.QtWidgets import QApplication
from src.graphic_interface import gui_single#import the main gui
import time

if __name__=="__main__":
    app = QApplication(sys.argv)
    gui = gui_single.gui_single(sys.argv[1])
    app.exec_()

