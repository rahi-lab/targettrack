#this handles the GUI
from . import gui_elements_plots as plots
from . import gui_elements_controls as controls
from . import image_rendering
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from ..helpers import QtHelpers

from PyQt5.QtGui import QKeySequence
from logging_config import setup_logger
logger = setup_logger(__name__)

class gui(QMainWindow):

    reserved_keys = ["z", "c", "a", "d", "v", "b", "n", "m", ",", ".", "\n", "", " "]

    def __init__(self, controller, settings, parent=None):
        super().__init__()
        self.controller = controller
        self.controller.freeze_registered_clients.append(self)
        self.settings = settings
        self.parent = parent

        self.setWindowTitle("Targettrack")
        self.resize(self.settings["screen_w"]*4//5, self.settings["screen_h"]*4//5)
        
        # Create central widget for the main figure
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # This is the holder for the main figure
        self.fig = plots.MainFigWidget(self.settings, self.controller, self.controller.frame_shape)
        main_layout.addWidget(self.fig)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        self.rendering = image_rendering.ImageRendering(self.controller, self.fig,
                                                    self.controller.data_name, self.controller.frame_num)

        # whether data is going to be as points or as masks:
        if self.controller.point_data is None:
            dtype = QtHelpers.DataTypeChoice.choose_data()
            if dtype == "points":
                self.controller.set_point_data(True)
            else:
                self.controller.set_point_data(False)

        # Create dockable neuron bar
        neuron_dock = QDockWidget("Neuron Bar", self)
        neuron_bar = controls.NeuronBar(self.controller, reserved_keys=self.reserved_keys)
        neuron_dock.setWidget(neuron_bar)
        self.addDockWidget(Qt.TopDockWidgetArea, neuron_dock)

        # Create dockable time slider
        slider_dock = QDockWidget("Time Control", self)
        slider = controls.TimeSlider(self.controller, self.controller.frame_num, int(self.settings["time_label_num"]))
        slider_dock.setWidget(slider)
        self.addDockWidget(Qt.BottomDockWidgetArea, slider_dock)

        # Create dockable tracks/activities tab
        tracks_dock = QDockWidget("Tracks & Activities", self)
        self.timedisplaytabs = QTabWidget()
        if True:
            self.trackstable = controls.DashboardTab(self.controller,
                                                   dashboard_chunk_size=int(self.settings["dashboard_chunk_size"]))
            self.calacts = plots.ActivityPlotWidget(self.controller,
                                                  int(self.settings["max_sim_tracks"]),
                                                  int(self.settings["tracks_num_row"]))
            self.timedisplaytabs.addTab(self.trackstable, "Tracks")
            self.timedisplaytabs.addTab(self.calacts, "Activity")
        tracks_dock.setWidget(self.timedisplaytabs)
        self.addDockWidget(Qt.RightDockWidgetArea, tracks_dock)

        # Create dockable control panel
        control_dock = QDockWidget("Controls", self)
        tracking_panel = QTabWidget()
        
        # View tab
        view_tab = controls.ViewTab(self.controller, self.rendering, self.settings)
        tracking_panel.addTab(view_tab, "View")
        tracking_panel.tabBar().setTabTextColor(0, QtGui.QColor(0,0,0))

        if True:
            # Annotation tab
            annotation_tab = controls.AnnotationTab(self.controller, self.controller.frame_num,
                                                self.settings["mask_threshold_for_new_region"])
            tracking_panel.addTab(annotation_tab, "Annotate")
            tracking_panel.tabBar().setTabTextColor(1, QtGui.QColor(0,0,0))

            # NN control tab
            NN_control_tab = controls.NNControlTab(self.controller, self.controller.data_name)
            tracking_panel.addTab(NN_control_tab, "NN")
            tracking_panel.tabBar().setTabTextColor(2, QtGui.QColor(0,0,0))

            if not self.controller.point_data:
                # Selection tab
                selection_tab = controls.SelectionTab(self.controller)
                tracking_panel.addTab(selection_tab, "Frame selection")
                tracking_panel.tabBar().setTabTextColor(3, QtGui.QColor(0,0,0))

                # Export/Import tab
                export_import_tab = controls.ExportImportTab(self.controller, self.controller.frame_num)
                tracking_panel.addTab(export_import_tab, "Export/Import")
                tracking_panel.tabBar().setTabTextColor(4, QtGui.QColor(0,0,0))

                # Processing tab
                processing_tab = controls.MaskProcessingTab(self.controller)
                tracking_panel.addTab(processing_tab, "Processing")
                tracking_panel.tabBar().setTabTextColor(5, QtGui.QColor(0,0,0))

            # IO tab
            IO_tab = controls.SavingTab(self.controller)
            tracking_panel.addTab(IO_tab, "IO")
            tracking_panel.tabBar().setTabTextColor(6, QtGui.QColor(0,0,0))

        control_dock.setWidget(tracking_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, control_dock)

        # Create dockable goto frame button
        goto_dock = QDockWidget("Go To Frame", self)
        goto_layout = controls.GoTo(self.controller, self.controller.frame_num)
        goto_dock.setWidget(goto_layout)
        self.addDockWidget(Qt.BottomDockWidgetArea, goto_dock)

        # Allow nested docks
        self.setDockNestingEnabled(True)

        # Allow docks to be tabified
        self.tabifyDockWidget(tracks_dock, control_dock)

        # these work wherever the mouse is, this is to move between times
        self.eventbucket={}
        for di,key in zip([-100,-10,-1,1,10,100],['v','b','n','m',',','.']):
            self.eventbucket[key]=QShortcut(QKeySequence(key), self)
            self.eventbucket[key].activated.connect(self._make_move_relative(di))
        self.eventbucket['a']=QShortcut(QKeySequence('a'), self)
        self.eventbucket['a'].activated.connect(self.controller.toggle_autocenter)
        self.eventbucket['c']=QShortcut(QKeySequence('c'), self)
        self.rotating=False
        self.eventbucket['c'].activated.connect(lambda:self.activate_rotate(abort=False))
        self.eventbucket['z']=QShortcut(QKeySequence('z'), self)
        self.eventbucket['z'].activated.connect(self.controller.undo_mask)
        self.eventbucket['\n']=QShortcut(QKeySequence('\n'), self)
        self.eventbucket['\n'].activated.connect(self.controller.flag_current_as_gt)
        self.eventbucket['x']=QShortcut(QKeySequence('x'), self)
        self.eventbucket['x'].activated.connect(lambda:self.activate_rotate(abort=True))

        # Save dock state
        self.settings["dock_state"] = self.saveState()

    def _make_move_relative(self, nb):
        def fun():
            self.controller.move_relative_time(nb)
        return fun

    def activate_rotate(self, abort=False):
        if abort:
            self.fig.rotator.hide()
            self.fig.rot_zero_ang()
            self.rotating=False
            return
        self.rotating=not self.rotating
        if self.rotating:
            self.fig.rotator.show()
        else:
            angle_txt = self.fig.rotator.angle()
            try:
                angle = float(angle_txt)
            except:
                pass
            else:
                self.controller.rotate_frame(angle)
            self.fig.rot_zero_ang()
            self.fig.rotator.hide()

    def freeze(self):
        self.setEnabled(False)

    def unfreeze(self):
        self.setEnabled(True)
    def closeEvent(self, event):
      self.parent.closeEvent(event)
      # self.close()
    def close(self,arg,msg):
        logger.debug("Closing GUI")
        ####Dependency
        ok,msg=self.controller.close(arg,msg)
        ####Self Behavior
        #this is equivalent to no behavior
        if not ok:
            return False,msg
        if arg=="force":
            return True,msg
        return True,msg

    #this is not used but will be used to manually re-center a frame
    def getRecenter(self,fn,dat,i,rsh):
        dialog = QDialog()
        label="Re-fetching frame from "+fn+" at time: "+str(i)
        pars=[None,None,None,None]
        lay=plots.RecenterPlotLayout(label, dat, rsh, pars, dialog)
        dialog.setLayout(lay)
        dialog.exec_()
        return pars
