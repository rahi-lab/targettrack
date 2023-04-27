import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFontMetrics


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
        self.controller.highlighted_neuron_registered_clients.append(self)

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

        self.autolevels = bool(int(self.settings["autolevels"]))

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
        self.pointsetplots = {}
        for key in pointsetnames:
            self.pointsetplots[key]=pg.ScatterPlotItem(pen=(self.pens[key] if key!="pts_act" else None),brush=(0,0,0,0))
            self.addItem(self.pointsetplots[key])
        self.pointsetsdata = {}   # will contain the x,y,z position of the points of each set.

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
                colors = self.controller.neuron_color()
                pens = [pg.mkPen(width=self.s_thick, color=color) for color in colors]   # Todo no need to create new pens every time
                self.pointsetplots[key].setData(pen=pens, pos=val[:, :2])
            else:
                self.pointsetplots[key].setData(pos=val[:, :2])
            self.pointsetplots[key].setSize(size=self.size_func(val[:,2]))
            self.pointsetsdata[key] = val

    def change_highlighted_neuron(self, high: int=None, unhigh:int=None, high_pointdat=None, **kwargs):
        """
        :param high_pointdat: 1x3 array with the x, y, z coordinates of the highlighted point.
            Must be given if high is not None (only in point_data mode).
        """
        high_key = "pts_high"
        if high is None:
            self.pointsetplots[high_key].setData(pos=[])
            self.pointsetsdata[high_key] = []
        elif high_pointdat is not None:
            self.pointsetplots[high_key].setData(pos=high_pointdat[:, :2])
            self.pointsetsdata[high_key] = high_pointdat
            self.pointsetplots[high_key].setSize(size=self.size_func(high_pointdat[:, 2]))

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
        new_z = max(-1, self.z + event.angleDelta().y()/8/15)
        self.controller.change_z(new_z)

    def change_z(self, value):
        if value == -1:
            self.z = value
        else:
            prop = int(np.clip(value, 0, self.img_data.shape[2]-1))
            if not np.isnan(prop):
                self.z = prop
        self.update_image_display()
        self.update_mask_display()
        self.label.setText(self.labeltext.format(self.z))
        if self.controller.point_data:
            for key, plot in self.pointsetplots.items():
                plot.setSize(self.size_func(self.pointsetsdata[key][:, 2]))

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
            if mask is False:   # in this case remove the mask
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


class ActivityPlotWidget(pg.PlotWidget,QGraphicsItem):
    """
    This displays one plot per neuron that has a key assigned.
    The plot is the calcium activity of the neuron on a range of times around current time.
    """
    def __init__(self, controller, max_sim_tracks:int, time_chunk_size):
        """
        :param controller: instance of Controller
        :param max_sim_tracks: the maximum number of neurons displayed
        """
        super().__init__()
        self.controller = controller
        self.controller.frame_registered_clients.append(self)
        self.controller.neuron_keys_registered_clients.append(self)
        self.controller.calcium_registered_clients.append(self)

        self.t = 0  # Todo good init
        self.nb_times = time_chunk_size
        self.nb_frames = self.controller.frame_num
        self.times = np.arange(self.nb_times)  # list of times in x-axis (for labels)   # TODO AD good init
        self.neuron_plotidx = {}  # dict neuron_idx_from1 -> idx such that self.plots[idx] corresponds to neuron neuron_idx_from1 at time t
        # Todo I think self.neuron_plotidx could be deleted

        self.setBackground('w')

        self.setLabel('left', "Intensity")
        self.setLabel('bottom', "Time[frames]")

        self.plots = []
        self.ebars = []
        self.neuron_activities = {}   # dict neuron_idx_from1 -> array of shape nb_frames * 2 with activity[t] is the
        # activity value and error bars of neuron neuron_id_from1 at time t

        for i in range(max_sim_tracks):
            self.plots.append(self.plot(antialias=True))
            self.ebars.append(pg.ErrorBarItem(antialias=True))#symbol="o"
            self.addItem(self.ebars[-1])
        self.timeline=pg.InfiniteLine(0,pen="r")
        self.addItem(self.timeline)

    def change_ca_activity(self, activity, neuron_id_from1=None):
        """
        Updates the activity plot of given neuron or of a given time.
        :param neuron_id_from1: standard neuron idx
        :param activity:
            if neuron_id_from1 is given: array of shape nb_frames * 2 with activity[tp] is the activity value and error
                bars of neuron neuron_id_from1 at time tp
            else: array of shape nb_neurons * nb_frames * 2, same as above for each neuron
            NOT in 1-indexing.
        """
        if neuron_id_from1 is not None:
            if neuron_id_from1 not in self.neuron_plotidx:
                return
            self.neuron_activities[neuron_id_from1] = activity
            self._update_neuron_plot(neuron_id_from1)
        else:
            for idx_from1 in self.neuron_activities:
                self.neuron_activities[idx_from1] = activity[idx_from1-1]
                self._update_neuron_plot(idx_from1)
        self.autoRange(items=self.plots)

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
            color = self.controller.neuron_color(neuron_id_from1)
            self.plots[ind].setData(x=self.times, y=yvals)
            self.plots[ind].setPen(width=2, color=color)   # Todo: no need to change pens all the time (esp when color has not changed)
            self.plots[ind].setSymbol("+")
            self.plots[ind].setSymbolPen(width=1, color=color)
            self.ebars[ind].setData(x=self.times, y=yvals, height=yerrs, pen={'color': color})

    def _remove_old_plots(self):
        """
        erases the plots of neurons that have been de-assigned, if there are fewer neurons than before
        """
        n_plots = len(self.neuron_plotidx)
        for ind in range(n_plots, len(self.plots)):
            self.plots[ind].setData()
            self.ebars[ind].setData()

    def change_t(self, t):
        """
        Callback when the current time changes.
        Updates the timeline, recomputes the times to display and updates the plot.
        """
        # first recompute times to display
        self.t = t
        box_ind = self.t // self.nb_times
        self.times = np.array([lab for lab in range(box_ind * self.nb_times, min((box_ind + 1) * self.nb_times, self.nb_frames))])

        # then update all plots...
        self.timeline.setValue(t)
        self._update_all_plots()

    def change_neuron_keys(self, changes: list):
        """
        Updates plots and tracks to take changes of assigned neurons into account.
        :param changes: list of (neuron_idx_from1, new_key). For each such pair, a plot and track will be displayed for
            neuron neuron_idx_from1 iff the key is not None.
        """
        neurons_to_display = set(self.neuron_plotidx.keys())
        add = []
        rm = []
        for i, (neuron_idx_from1, key) in enumerate(changes):
            if key is None:
                neurons_to_display.discard(neuron_idx_from1)
                rm.append(neuron_idx_from1)
            elif neuron_idx_from1 not in neurons_to_display:
                neurons_to_display.add(neuron_idx_from1)
                add.append(neuron_idx_from1)
        if add or rm:   # if not, then the set of neurons to display has not changed.
            self.neuron_plotidx = dict(zip(sorted(neurons_to_display), range(len(neurons_to_display))))
            if rm:
                for neuron_idx_from1 in rm:
                    if neuron_idx_from1 in self.neuron_activities:
                        del self.neuron_activities[neuron_idx_from1]
                self._remove_old_plots()
            if add:
                for neuron_idx_from1 in add:
                    self.neuron_activities[neuron_idx_from1] = self.controller.neuron_ca_activity(neuron_idx_from1)
            self._update_all_plots()
