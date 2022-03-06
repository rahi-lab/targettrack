#this handles the GUI
from . import gui_elements_plots as plots
from . import gui_elements_controls as controls
from . import image_rendering
from PyQt5.QtWidgets import *
from PyQt5 import QtGui

from ..helpers import QtHelpers

from PyQt5.QtGui import QKeySequence


class gui(QMainWindow):

    reserved_keys = ["z", "c", "a", "d", "v", "b", "n", "m", ",", ".", "\n", "", " "]

    def __init__(self, gui, controller, windowsize):
        #we initialize this as a Widget
        super().__init__()
        self.gui= gui   # Todo AD not sure this is meant to stay, but for now I need to have a controller available right from the beginning
        self.controller = controller
        self.settings = gui.settings

        self.setWindowTitle("Tracking")
        self.resize(self.gui.settings["screen_w"]*4//5, self.gui.settings["screen_h"]*4//5)
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        tracking_grid = QGridLayout()#this is the full grid
        topgrid=QGridLayout()

        tracking_grid_left = QGridLayout()

        tracking_grid_right = QGridLayout()



        # this is the neuron bar at the top.
        neuron_bar = controls.NeuronBar(self.controller, reserved_keys=self.reserved_keys)
        tracking_grid_left.addWidget(neuron_bar, 0, 0)#, 1, 2)


        # This is the holder for the main figure. MainFigLayout.
        self.fig = plots.MainFigWidget(self.settings, self.controller, self.controller.frame_shape)
        tracking_grid_left.addWidget(self.fig, 1, 0)#, 1, 2)
        self.rendering = image_rendering.ImageRendering(self.controller, self.fig,
                                                        self.controller.data_name, self.controller.frame_num)

        # Time slider at the bottom
        slider = controls.TimeSlider(self.controller, self.controller.frame_num, int(self.settings["time_label_num"]))
        tracking_grid_left.addWidget(slider, 2, 0)

        tracking_grid_left.setRowStretch(0,0)
        tracking_grid_left.setRowStretch(1,1)
        tracking_grid_left.setRowStretch(2,0)

        # This is the tab of Tracks and Activities
        self.timedisplaytabs = QTabWidget()
        if True:
            self.trackstable = controls.DashboardTab(self.controller,
                                                     dashboard_chunk_size=int(self.settings["dashboard_chunk_size"]))
            self.calacts = plots.ActivityPlotWidget(self.controller,
                                                    int(self.settings["max_sim_tracks"]),
                                                    int(self.settings["tracks_num_row"]))
            self.timedisplaytabs.addTab(self.trackstable, "Tracks")
            self.timedisplaytabs.addTab(self.calacts, "Activity")

        # Now add the tab to the right grid above
        tracking_grid_right.addWidget(self.timedisplaytabs, 0, 0)

        #this is the control panel with many features
        tracking_panel = QTabWidget()#this is self.plotstab in Core's code
        # this takes care of the image rendering
        view_tab = controls.ViewTab(self.controller, self.rendering, self.settings)

        #tracking_panel.setFixedSize(view_tab.sizeHint().width(),self.gui.settings["screen_h"]//3)

        tracking_panel.addTab(view_tab, "View")
        tracking_panel.tabBar().setTabTextColor(0,QtGui.QColor(0,0,0))
        if True:
            #tracking_panel.setStyleSheet("QTabBar::tab { height: 30px; width: 100px}")

            # this is tools to annotate the data
            annotation_tab = controls.AnnotationTab(self.controller, self.controller.frame_num,
                                                    self.settings["mask_threshold_for_new_region"])
            tracking_panel.addTab(annotation_tab, "Annotate")
            tracking_panel.tabBar().setTabTextColor(1,QtGui.QColor(0,0,0))

            # this manages the NNs
            NN_control_tab = controls.NNControlTab(self.controller, self.controller.data_name)
            
            tracking_panel.addTab(NN_control_tab, "NN")
            tracking_panel.tabBar().setTabTextColor(2,QtGui.QColor(0,0,0))

            # this is to select subsets of the data
            selection_tab = controls.SelectionTab(self.controller)
            tracking_panel.addTab(selection_tab, "Frame selection")
            tracking_panel.tabBar().setTabTextColor(3,QtGui.QColor(0,0,0))

            # this is the preprocessing tab
            preprocessing_tab = controls.PreProcessTab(self.controller,self.controller.frame_num,
                                                    self.settings["mask_threshold_for_new_region"])

            tracking_panel.setFixedSize(view_tab.sizeHint().width(),self.gui.settings["screen_h"]//3)

            tracking_panel.addTab(preprocessing_tab, "Export/Import")
            tracking_panel.tabBar().setTabTextColor(4,QtGui.QColor(0,0,0))

            # this is the mask processes
            processing_tab = controls.MaskProcessingTab(self.controller)
            tracking_panel.addTab(processing_tab, "Processing")
            tracking_panel.tabBar().setTabTextColor(5,QtGui.QColor(0,0,0))

            # this is for saving
            IO_tab = controls.SavingTab(self.controller)
            tracking_panel.addTab(IO_tab, "IO")
            tracking_panel.tabBar().setTabTextColor(6,QtGui.QColor(0,0,0))

        tracking_grid_right.addWidget(tracking_panel,1,0)


        # Todo: do we not want the slider in tracking_grid_left rather than tracking_grid?

        # This is the goto frame button
        goto_layout = controls.GoTo(self.controller, self.controller.frame_num)
        tracking_grid_right.addWidget(goto_layout, 2, 0)

        tracking_grid.addLayout(topgrid,0,0)
        tracking_grid.addLayout(tracking_grid_left,1,0)
        tracking_grid.addLayout(tracking_grid_right,1,1)
        tracking_grid.setColumnStretch(0,5)
        tracking_grid.setColumnStretch(1,1)

        self.centralWidget.setLayout(tracking_grid)



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

        # whether data is going to be as points or as masks:
        if self.controller.point_data is None:
            dtype = QtHelpers.DataTypeChoice.choose_data()
            if dtype == "points":
                self.controller.set_point_data(True)
            else:
                self.controller.set_point_data(False)

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


    def close(self,arg,msg):
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
