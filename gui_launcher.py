#this is the main wrapper
import sys
assert len(sys.argv)==2, "No dataset name given"#Check that the dataset name is given
from PyQt5.QtWidgets import QApplication
from src.graphic_interface import gui_single#import the main gui
import time

app = QApplication(sys.argv)

gui = gui_single.gui_single(sys.argv[1])

app.exec_()

try:
    assert False
    app.exec_()
except:
    print("This will be the message in practice:")
    print(sys.exc_info())
    print("If you were training a network, it might be running in the background. Sorry...")
    print("Please Report this Bug(and how to reproduce it!) to cfpark00@gmail.com")
