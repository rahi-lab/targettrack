import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFontMetrics


class NearFigWidget(pg.PlotWidget,QGraphicsItem):
    def __init__(self,settings,controller,rel_t):
        super().__init__()
        self.setMenuEnabled(False)
        #self.invertY(True)
        self.setAspectLocked(True)
        self.setContentsMargins(0,0,0,0)

        self.label = ""

        self.scene().sigMouseClicked.connect(self.mouseclick)

        self.controller = controller

        self.settings=settings

        self.img = pg.ImageItem(np.zeros((100,100,3)))
        self.addItem(self.img)
        self.hideAxis('bottom')
        self.hideAxis('left')
        self.labeltext=""
        self.label=pg.TextItem(self.labeltext)
        self.addItem(self.label)

        self.inn=True
        self.rel_t=rel_t


    def mouseclick(self,event):
        self.controller.move_relative_time(self.rel_t)

    def enterEvent(self, QEvent):
        self.setFocus()
        self.inn=True

    def leaveEvent(self, QEvent):
        self.inn=False

    def update_data(self,dat):
        #unpacks the data struct and update all the corresponding plots
        for key,val in dat.items():
            if key=="img":
                self.imdat=val
            if key=="label":
                self.labeltext=val
        self.update()

    def update(self):
        #when z updates the sizes changes
        self.img.setImage(self.imdat)
        self.label.setText(self.labeltext)


# this is the main figure
class MainFigWidget(pg.PlotWidget,QGraphicsItem):

    def __init__(self,settings, controller, frame_shape):
        """
        Creates the gui main plot, which displays the frame, possibly masks or points, label...
        :param settings:
        :param controller: the instance of Controller this figure reports to
        :param frame_shape: the 3D shape of a frame (height*width*depth)
        """
        super().__init__()

        self.setMenuEnabled(False)
        self.invertY(False)
        self.setAspectLocked(True)

        self.scene().sigMouseMoved.connect(self.mousehandle)
        self.scene().sigMouseClicked.connect(self.mouseclick)

        self.settings=settings
        self.controller = controller
        self.controller.points_registered_clients.append(self)
        self.controller.pointlinks_registered_clients.append(self)
        self.controller.highlighted_track_registered_clients.append(self)
        self.controller.zslice_registered_clients.append(self)
        #self.controller.highlighted_neuron_registered_clients.append(self) MB:should this be added

        # image holder for the video frame
        self.img = pg.ImageItem(np.zeros((100,100,3)))
        self.addItem(self.img)
        # image holder for the masks (if there are none, will be fully transparent)
        self.mask_img = pg.ImageItem(np.zeros((100, 100, 4)))
        self.addItem(self.mask_img)

        self.img_data = None
        self.mask_data = None

        self.hideAxis('bottom')
        self.hideAxis('left')

        self.labeltext=""
        self.label=pg.TextItem(self.labeltext)
        self.addItem(self.label)

        self.shape = frame_shape
        self.zmax= self.shape[2] - 1
        self.z=(self.zmax+1)//2

        self.autolevels = True   # TODO AD: set good default? (or set with settings)

        self.min_s_size=int(self.settings["min_s_size"])
        self.max_s_size=int(self.settings["max_s_size"])
        self.s_thick=float(self.settings["s_thick"])
        self.s_z_wid=int(self.settings["s_z_wid"])
        #this simple returns the radial sizes of the annotations depending on the z separation. Triangular scheme -^-
        def size_func(z_in):
            return np.clip(self.min_s_size+(self.s_z_wid-np.abs(z_in-self.z))*(self.max_s_size/self.s_z_wid),self.min_s_size,self.max_s_size)

        self.size_func=size_func

        # Set up displaying points
        pointsetnames=["pts_pointdat","pts_NN_pointdat","pts_adj","pts_high","pts_act"]#,"pts_bayes"
        self.pens={}
        self.pens["pts_pointdat"]=pg.mkPen(width=self.s_thick, color=pg.mkColor(255,255,255,128))
        self.pens["pts_NN_pointdat"]=pg.mkPen(width=self.s_thick, color=pg.mkColor(255,0,25,128))
        #self.pens["pts_bayes"]=pg.mkPen(width=self.s_thick, color=pg.mkColor(255,100,100,128))
        self.pens["pts_adj"]=pg.mkPen(width=self.s_thick, color=pg.mkColor(0,255,25,128))
        self.pens["pts_high"]=pg.mkPen(width=self.s_thick+1, color=pg.mkColor(255,255,255,255))
        actcolors=[(31, 119, 180),(255, 127, 14),(44, 160, 44),(214, 39, 40),(148, 103, 189),(140, 86, 75),(227, 119, 194),(127, 127, 127),(188, 189, 34),(23, 190, 207)]
        self.actpens=[pg.mkPen(width=self.s_thick, color=color) for color in actcolors]
        self.pointsetplots = {}
        for key in pointsetnames:
            self.pointsetplots[key]=pg.ScatterPlotItem(pen=(self.pens[key] if key!="pts_act" else None),brush=(0,0,0,0))
            self.addItem(self.pointsetplots[key])

        self.linkplot=pg.GraphItem()
        self.addItem(self.linkplot)
        self.rotator=pg.EllipseROI([self.shape[0] / 4, self.shape[1] / 4], [self.shape[0] / 2, self.shape[1] / 2], pen=(1, 9), movable=False)
        self.addItem(self.rotator)
        self.rotator.removeHandle(1)
        self.rotator.hide()

        self.track=pg.PlotCurveItem()
        self.addItem(self.track)
        self.inn=True

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


    def rot_zero_ang(self):
        self.rotator.setAngle(0,center=[0.5,0.5])

    def mousehandle(self,event):
        self.mouseev=event

    def mouseclick(self,event):
        if self.z==-1:
            return
        pos=self.plotItem.vb.mapSceneToView(event.scenePos())
        self.controller.frame_clicked(event.button(), [pos.x(),pos.y(),self.z])

    def keyPressEvent(self, event):
        if event.text()==" ":
            self.plotItem.vb.enableAutoRange()
            return
        if self.z==-1:
            return
        if self.inn:
            pos=self.plotItem.vb.mapSceneToView(self.mouseev)
            self.controller.key_pressed(event.text(),[pos.x(),pos.y(),self.z])

    def enterEvent(self, QEvent):
        self.setFocus()
        self.inn=True

    def leaveEvent(self, QEvent):
        self.inn=False

    def change_pointdats(self, pts_dict:dict):
        """
        Updates the points data and display
        :param pts_dict: dict str -> array of shape neurons*3, assigning to a type of points the xyz
            coordinates of the point of each neuron
        """
        for key, val in pts_dict.items():
            if key == "pts_act":
                pens = [self.actpens[i] for i in range(len(val))]
                self.pointsetplots[key].setData(pen=pens, pos=val[:, :2])
            else:
                self.pointsetplots[key].setData(pos=val[:, :2])
            self.pointsetplots[key].setSize(size=self.size_func(val[:,2]))


    def change_links(self, link_data):
        """
        Updates the displayed links between points
        :param link_data: None if no links to display; otherwise a 2D array of size (2nb_links)*3
        """
        if link_data is None:   # remove links
            self.linkplot.setData(pos=np.zeros((0, 2)), adj=np.zeros((0, 2), dtype=np.int8))
        else:
            n_link = len(link_data) // 2
            adj = np.array([np.arange(n_link), np.arange(n_link) + n_link]).T
            self.linkplot.setData(pos=link_data, adj=adj, pen="w", symbol=None)

    def change_track(self, track):
        """ Updates the track of the highlighted neuron. """
        self.track.setData(track[0], track[1])

    def wheelEvent(self,event):
        self.change_z(max(-1, self.z + event.angleDelta().y()/8/15))

    def change_z(self, value):
        # TODO if any other classes need to use z, then wheelEvent should only notify controller of z change and let the
        #  subsequent call to self.change_z do the job
        if value == -1:
            self.z = value
        else:
            prop = int(np.clip(value, 0, self.img_data.shape[2]-1))   # Todo: could be a pb if img_data does not yet exist
            if not np.isnan(prop):
                self.z = prop
        self.update_image_display()
        self.update_mask_display()
        self.label.setText(self.labeltext.format(self.z))

    def set_data(self, img=None, mask=None, label=None):
        """
        Change the data to display (the image, mask, and label can be changed independently) and update the display accordingly.
        :param img: the (3D) image to display
        :param mask: the (3D) mask to display. Give None if not changing, give False if removing mask, give new mask to change.
        :param label: label to display. Must contain one {} field, to be filled by z.
        """
        if img is not None:
            self.img_data = img
            self.update_image_display()

        if mask is not None:
            if mask is False:   # in this case reomve the mask
                self.mask_data = None
            else:
                self.mask_data = mask
            self.update_mask_display()

        if label is not None:
            self.labeltext = label
            self.label.setText(self.labeltext.format(self.z))

    def update_image_display(self):
        """
        Updates the display of the image
        """
        # SJR: Figure out which image (z-slice or maximum intensity projection)
        if self.z == -1:
            img = np.max(self.img_data, axis=2)
        else:
            img = self.img_data[:, :, self.z]

        self.img.setImage(img, autoLevels=self.autolevels,
                          levels=(0, 1))  # Todo: verify that levels does not override autoLevels

    def update_mask_display(self):
        """
        Updates the display of the mask
        """
        # SJR: Only deal with mask if there is one
        if self.mask_data is None:
            self.mask_img.clear()
        else:
            # SJR: Figure out which image (z-slice or z-average of mask colors)
            if self.z == -1:
                mask = np.max(self.mask_data, axis=2)#MB changes mean to max
            else:
                mask = self.mask_data[:, :, self.z].copy()
            self.mask_img.setImage(mask)


# This plots the tracks
class TracksTable(QWidget):
    """The tracks"""

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']

    def __init__(self, parent, controller,  tracks_cell_height:int):
        """
        :param parent: instance of TimeDisplay in which self is embedded
        :param tracks_cell_height: fixed height of the cell
        """
        super().__init__()
        self.parent = parent
        self.controller = controller
        self.controller.highlighted_neuron_registered_clients.append(self)
        #self.controller.points_registered_clients.append(self)#MB removed this to prevent error. where do we need poin update in this widget?

        self.tracks_cell_height = tracks_cell_height

        # This is a mess. It handles the tracks table so that the very top and left columns are always visible
        self.tracks_full_lay = QGridLayout()
        self.tracks_full_lay.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.tracks_full_lay)

        self.tracksgrid_top_holder = QVBoxLayout()
        self.tracksgrid_top_holder.setContentsMargins(5, 5, 40, 5)
        self.tracksgrid_top = QGridLayout()
        self.tracksgrid_top.setContentsMargins(0, 0, 0, 0)

        self.tracksgrid_top_holder.addLayout(self.tracksgrid_top)

        self.tracks_scroll = QScrollArea()
        self.tracks_scroll.setWidgetResizable(True)
        self.tracks_scroll.setContentsMargins(0, 0, 0, 0)

        dummy = QWidget()

        self.tracksgrid_holder = QVBoxLayout()
        self.tracksgrid_holder.setContentsMargins(5, 5, 25, 5)
        self.tracksgrid = QGridLayout()
        self.tracksgrid.setContentsMargins(0, 0, 0, 0)
        self.tracksgrid_holder.addLayout(self.tracksgrid)

        dummy.setLayout(self.tracksgrid_holder)
        self.tracks_scroll.setWidget(dummy)

        self.tracks = {}
        self.tracks_col_labels = {}
        self.tracks_time_labels = []

        # initialize the time labels
        for j in range(len(self.times)):
            t = QPushButton(" ")
            t.clicked.connect(self._make_goto_fun(j))
            t.setStyleSheet("height: 10px; width: 10px;min-width: 10px;")
            self.tracks_time_labels.append(t)

        self.have_tracks = False

        self.tracks_full_lay.addLayout(self.tracksgrid_top_holder, 0, 0)
        self.tracks_full_lay.addWidget(self.tracks_scroll, 1, 0)

    def _make_goto_fun(self, j):
        def fun():
            self._go_to_frame(j)
        return fun

    def _go_to_frame(self, j):
        self.controller.go_to_frame(j + self.nb_times * self.parent.time_box())

    def _create_track(self):
        tracks_lst = []
        for j in range(self.nb_times):
            temp = QWidget()
            temp.setFixedHeight(self.tracks_cell_height)
            temp2 = QHBoxLayout()
            temp2.setContentsMargins(0, 0, 0, 0)

            temp3 = QPushButton(" ")
            temp3.setStyleSheet("background-color: red; height: 10px; width: 10px;min-width: 10px;")
            temp2.addWidget(temp3)

            temp.setLayout(temp2)
            temp.pointer_to_button = temp3
            tracks_lst.append(temp)
        return tracks_lst

    def _create_neuron_label(self, neuron_idx_from1):
        t = QPushButton(" ")
        t.clicked.connect(lambda: self.controller.highlight_neuron(neuron_idx_from1))   # TODO AD
        return t

    @property
    def times(self):
        return self.parent.times

    @property
    def nb_times(self):
        return self.parent.nb_times

    @property
    def neuron_plotidx(self):
        return self.parent.neuron_plotidx

    def change_present_neurons(self, present=None, added=None, removed=None):
        """
        Changes which of the neurons are present in current frame, as their corresponding buttons should be colored
        in green instead of red.
        :param present: which neuron indices (from 1) are present, if given
        :param added: single neuron index (from 1) that was added, if given
        :param removed: single neuron index (from 1) that was removed, if given
        """
        crow = self.parent.time_idx()
        if present is not None:
            for ind in self.neuron_plotidx.values():   # first reset all as absent
                self.tracks[ind][crow].pointer_to_button.setStyleSheet(
                    "background-color: red; height: 10px; width: 10px;min-width: 10px;")
            for neuron_id_from1 in present:   # then set present as such
                ind = self.neuron_plotidx[neuron_id_from1]
                self.tracks[ind][crow].pointer_to_button.setStyleSheet(
                    "background-color: green; height: 10px; width: 10px;min-width: 10px;")
            return
        if added in self.neuron_plotidx:   # None will not be in self.neuron_plotidx
            ind = self.neuron_plotidx[added]
            self.tracks[ind][crow].pointer_to_button.setStyleSheet(
                "background-color: green; height: 10px; width: 10px;min-width: 10px;")
        if removed in self.neuron_plotidx:   # None will not be in self.neuron_plotidx
            ind = self.neuron_plotidx[removed]
            self.tracks[ind][crow].pointer_to_button.setStyleSheet(
                "background-color: red; height: 10px; width: 10px;min-width: 10px;")

    def change_highlighted_neuron(self, high: int=None, unhigh:int=None, **kwargs):
        """
        Highlights or unhighlights neuron buttons.
        :param high: neuron id (from 1), will be highlighted if given
        :param unhigh: neuron id (from 1), will be unhighlighted if given
        """
        if unhigh in self.neuron_plotidx:
            # ind = self.neuron_plotidx[unhigh]
            for row in range(self.nb_times):
                self.tracks[unhigh][row].setStyleSheet("height: 10px; width: 10px;min-width: 10px;")
                # self.tracksgrid.itemAtPosition(row+1,ind+1).widget().setStyleSheet("height: 10px; width: 10px;min-width: 10px;")
        if high in self.neuron_plotidx:
            # ind = self.neuron_plotidx[high]
            for row in range(self.nb_times):
                self.tracks[high][row].setStyleSheet("background-color: orange; height: 10px; width: 10px;min-width: 10px;")
                # self.tracksgrid.itemAtPosition(row+1,ind+1).widget().setStyleSheet("background-color: orange; height: 10px; width: 10px;min-width: 10px;")

    def t_box_changed(self):

        self.tracks_time_labels[-1].setText(str(self.times[-1]))
        fm = QFontMetrics(self.tracks_time_labels[-1].font())
        w = fm.width(self.tracks_time_labels[-1].text())
        self.tracks_pivot_holder.setFixedHeight(self.tracks_cell_height)
        self.tracks_pivot_holder.setFixedWidth(w)
        for j, lab in enumerate(self.times):
            self.tracks_time_labels[j].setText(str(lab))
            self.tracks_time_labels[j].setFixedHeight(self.tracks_cell_height)
            self.tracks_time_labels[j].setFixedWidth(w)
            self.tracksgrid.addWidget(self.tracks_time_labels[j], j + 1, 0)
        for ext in range(len(self.times), self.nb_times):   # TODO AD: wtf?? could mean no labels at end if t is close to frame_num
            self.tracks_time_labels[ext].setText("None")
            self.tracks_time_labels[ext].setFixedHeight(self.tracks_cell_height)
            self.tracks_time_labels[ext].setFixedWidth(w)
            self.tracksgrid.addWidget(self.tracks_time_labels[ext], ext + 1, 0)

        for j, t in enumerate(self.times):
            present_neurons = set(self.controller.present_neurons_at_t(t))
            for neuron_idx_from1 in self.neuron_plotidx:
                if neuron_idx_from1 in present_neurons:
                    self.tracks[neuron_idx_from1][j].pointer_to_button.setStyleSheet(
                        "background-color: green; height: 10px; width: 10px;min-width: 10px;")
                else:
                    self.tracks[neuron_idx_from1][j].pointer_to_button.setStyleSheet(
                        "background-color: red; height: 10px; width: 10px;min-width: 10px;")

    # TODO AD: define change_present_neurons   (but what about when present neurons in another frame displayed here change??)

    def t_row_changed(self, old_t):
        old_crow = self.parent.time_idx(old_t)
        self.trackstable.tracks_time_labels[old_crow].setStyleSheet("height: 10px; width: 10px;min-width: 10px;")
        crow = self.parent.time_idx()
        self.tracks_time_labels[crow].setStyleSheet(
            "background-color: orange; height: 10px; width: 10px;min-width: 10px;")
        self.tracks_scroll.ensureWidgetVisible(self.tracksgrid.itemAtPosition(crow + 1, 0).widget())

    def neuron_keys_changed(self, changes: list, add=None, rm=None, old_neurons=None):
        """
        Updates tracks to take changes of assigned neurons into account.
        :param changes: list of (neuron_idx_from1, new_key). For each such pair, a plot and track will be displayed for
            neuron neuron_idx_from1 iff the key is not None.
        :param add, rm: list of neuron indices (from1) of neurons newly plotted and removed from plots, respectively.
        """
        self.tracksgrid.setContentsMargins(0, 0, 0, 0)

        # remove all labels and tracks from display
        for neuron_idx_from1 in old_neurons:
            self.tracksgrid_top.removeWidget(self.tracks_col_labels[neuron_idx_from1])
            for track in self.tracks[neuron_idx_from1]:
                self.tracksgrid.removeWidget(track)

        # delete label and tracks of removed neurons
        for neuron_idx_from1 in rm:
            del self.tracks[neuron_idx_from1]
            del self.tracks_col_labels[neuron_idx_from1]

        # create new label and tracks for added neurons
        for neuron_idx_from1 in add:
            self.tracks[neuron_idx_from1] = self._create_track()
            if neuron_idx_from1 == self.controller.highlighted:
                for track in self.tracks[neuron_idx_from1]:
                    track.setStyleSheet("background-color: orange; height: 10px; width: 10px;min-width: 10px;")
            self.tracks_col_labels[neuron_idx_from1] = self._create_neuron_label(neuron_idx_from1)
        for j, t in enumerate(self.times):
            present_neurons = set(self.controller.present_neurons_at_t(t))
            for neuron_idx_from1 in add:
                if neuron_idx_from1 in present_neurons:
                    bg_color = "green"
                else:
                    bg_color = "red"
                for track in self.tracks[neuron_idx_from1]:
                    track.pointer_to_button.setStyleSheet(
                        "background-color: {}; height: 10px; width: 10px;min-width: 10px;".format(bg_color))

        # set the right label for changed neurons
        for neuron_idx_from1, key in changes:
            if key is not None:
                self.tracks_col_labels[neuron_idx_from1].setText(key)

        # now put all labels and tracks into the display
        for neuron_idx_from1, i in self.neuron_plotidx.items():
            for row in range(self.nb_times):
                self.tracksgrid.addWidget(self.tracks[neuron_idx_from1][row], row + 1, i + 1)
            self.tracksgrid_top.addWidget(self.tracks_col_labels[neuron_idx_from1], 0, i + 1)
            self.tracks_col_labels[neuron_idx_from1].setStyleSheet(
                "background-color: " + self.colors[i % 10] + "; height: 10px; width: 10px;min-width: 10px;")

        self.have_tracks = True



class ActivityPlotWidget(pg.PlotWidget,QGraphicsItem):
    """
    This displays one plot per neuron that has a key assigned.
    The plot is the calcium activity of the neuron on a range of times around current time.
    """
    def __init__(self, parent, controller, max_sim_tracks:int):
        """
        :param parent: instance of TimeDisplay in which self is embedded
        :param controller: instance of Controller
        :param max_sim_tracks: the maximum number of neurons displayed
        """
        super().__init__()
        self.parent = parent
        self.controller = controller
        self.controller.calcium_registered_clients.append(self)

        self.setBackground('w')

        self.setLabel('left', "Intensity")
        self.setLabel('bottom', "Time[frames]")

        #Same as MainFigWidget
        actcolors=[(31, 119, 180),(255, 127, 14),(44, 160, 44),(214, 39, 40),(148, 103, 189),(140, 86, 75),(227, 119, 194),(127, 127, 127),(188, 189, 34),(23, 190, 207)]
        self.actpens=[pg.mkPen(width=2, color=color) for color in actcolors]


        self.plots = []
        self.ebars = []
        self.neuron_activities = {}   # dict neuron_idx_from1 -> array of shape nb_frames * 2 with activity[t] is the
        # activity value and error bars of neuron neuron_id_from1 at time t

        for i in range(max_sim_tracks):
            self.plots.append(self.plot(pen=self.actpens[i],antialias=True))
            self.ebars.append(pg.ErrorBarItem(pen=self.actpens[i],antialias=True))#symbol="o"
            self.addItem(self.ebars[-1])
        self.timeline=pg.InfiniteLine(0,pen="r")
        self.addItem(self.timeline)

    @property
    def times(self):
        return self.parent.times

    @property
    def neuron_plotidx(self):
        return self.parent.neuron_plotidx

    def change_ca_activity(self, neuron_id_from1, activity):
        """
        Updates the activity plot of given neuron
        :param neuron_id_from1: standard neuron idx
        :param activity: array of shape nb_frames * 2 with activity[t] is the activity value and error bars of neuron
            neuron_id_from1 at time t
        """
        if neuron_id_from1 not in self.neuron_plotidx:
            return
        self.neuron_activities[neuron_id_from1] = activity
        self._update_neuron_plot(neuron_id_from1)
        self.autoRange(items=self.plots)

    def displayed_neurons_changed(self, add=None, rm=None):
        if add:
            for neuron_idx_from1 in add:
                self.neuron_activities[neuron_idx_from1] = self.controller.neuron_ca_activity(neuron_idx_from1)
        if rm:
            for neuron_idx_from1 in add:
                del self.neuron_activities[neuron_idx_from1]
        self._update_all_plots()

    def _update_all_plots(self):
        """Updates all plots, given that data and plot indices are correct."""
        for neuron_idx_from1 in self.neuron_plotidx:
            self._update_neuron_plot(neuron_idx_from1)
        self.autoRange(items=self.plots)

    def _update_neuron_plot(self, neuron_id_from1):
        """Updates the plot for neuron neuron_id_from1, given that data and plot indices are correct."""
        ind = self.neuron_plotidx[neuron_id_from1]
        activity = self.neuron_activities[neuron_id_from1][self.times]
        if np.sum(~np.isnan(activity[:, 0])) == 0:  # if all activities are nan, empty plots
            self.plots[ind].setData()
            self.ebars[ind].setData()
        else:
            scale = np.nanmax(activity[:, 0])
            yvals = (activity[:, 0] / scale + ind)
            yerrs = activity[:, 1] / scale
            self.plots[ind].setData(x=self.times, y=yvals)
            self.ebars[ind].setData(x=self.times, y=yvals, height=yerrs)

    def t_changed(self, t):
        self.timeline.setValue(t)
        self._update_all_plots()


class TimeDisplay(QTabWidget):
    """
    This contains the types of time evolution graphs: tracks and calcium activities
    """
    def __init__(self, controller, max_sim_tracks:int, nb_times:int, nb_frames:int, tracks_cell_height:int):
        """
        :param controller: instance of Controller
        :param max_sim_tracks: the maximum number of neurons displayed
        :param nb_times: the number of times on the x-axis
        :param nb_frames: number of frames in video
        :param tracks_cell_height: fixed height of the cell
        """
        super().__init__()
        self.controller = controller
        self.controller.frame_registered_clients.append(self)
        self.controller.neuron_keys_registered_clients.append(self)

        self.nb_times = nb_times
        self.nb_frames = nb_frames

        self.t = 0   # TODO AD good init
        self.times = list(range(self.nb_times))   # list of times in x-axis (for labels)   # TODO AD good init
        self.neuron_plotidx = {}   # dict neuron_idx_from1 -> idx such that self.calacts.plots[idx] and
        # self.tracksgrid.itemAtPosition(t+1,idx+1) correspond to neuron neuron_idx_from1 at time t

        self.trackstable = TracksTable(self, self.controller, tracks_cell_height)
        self.calacts = ActivityPlotWidget(self, self.controller, max_sim_tracks)
        self.addTab(self.trackstable, "Tracks")
        self.addTab(self.calacts, "Activity")



    def change_t(self, t):
        """
        Callback when the current time changes.
        Updates the timeline, recomputes the times to display and updates the plot.   # TODO AD
        """
        # first recompute times to display
        old_t = self.t
        self.t = t
        box_ind = self.time_box()
        self.times = [lab for lab in range(box_ind * self.nb_times, min((box_ind + 1) * self.nb_times, self.nb_frames))]

        # then update all plots...
        self.calacts.t_changed(t)

        # ... and all tracks
        if not self.trackstable.have_tracks:
            return
        if self.time_box(old_t) != box_ind:
            self.trackstable.t_box_changed()
        self.trackstable.t_row_changed(old_t)   # TODO: could be avoided if not actually changed??

    def change_neuron_keys(self, changes: list):
        """
        Updates plots and tracks to take changes of assigned neurons into account.
        :param changes: list of (neuron_idx_from1, new_key). For each such pair, a plot and track will be displayed for
            neuron neuron_idx_from1 iff the key is not None.
        """
        old_neurons = list(self.neuron_plotidx.keys())
        neurons_to_display = set(self.neuron_plotidx.keys())
        add = []
        rm = []
        for i, (neuron_idx_from1, key) in enumerate(changes):
            if key is None:
                neurons_to_display.discard(neuron_idx_from1)
                rm.append(neuron_idx_from1)
            elif neuron_idx_from1 not in self.neuron_plotidx:
                neurons_to_display.add(neuron_idx_from1)
                add.append(neuron_idx_from1)
        if add or rm:   # if not, then the set of neurons to display has not changed.
            self.neuron_plotidx = dict(zip(sorted(neurons_to_display), range(len(neurons_to_display))))
            self.calacts.displayed_neurons_changed(add=add, rm=rm)
        self.trackstable.neuron_keys_changed(changes, add=add, rm=rm, old_neurons=old_neurons)

    def time_idx(self, t=None):
        if t is None:
            t = self.t
        return self.t % self.nb_times

    def time_box(self, t=None):
        if t is None:
            t = self.t
        return t // self.nb_times
