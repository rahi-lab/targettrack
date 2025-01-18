"""
This is the core of the data-to-GUI link.
This contains a single class: Tracking
The main method is respond where it receives signals from the GUI, and performs a task.
"""

#External Packages
import warnings
import numpy as np
import os
import glob
import skimage.feature as skfeat
import scipy.spatial as spat
import scipy.ndimage as sim
import cv2
import threading
#Internal classes
from .helpers import SubProcManager, QtHelpers, misc
from . import h5utils
from .datasets_code.DataSet import DataSet
import shutil

#HarvardLab specific classes
from .calcium_activity import HarvardLab

# EPFL lab specific processing and data structures
from .parameters.GlobalParameters import GlobalParameters
from .mask_processing.segmentation import Segmenter
from .mask_processing.features import FeatureBuilder
from .mask_processing.clustering import Clustering
from .mask_processing.classification import Classification
from .mask_processing.image_register import Register_Rotate
from .mask_processing.NN_related import post_process_NN_masks, post_process_NN_masks2, post_process_NN_masks3, \
    post_process_NN_masks4, post_process_NN_masks5
from .mask_processing.image_processing import blur, blacken_background, resize_frame

# SJR: message box for indicating neuron number of new neuron and for renumbering neuron
from .msgboxes import EnterCellValue as ecv
from logging_config import setup_logger
logger = setup_logger(__name__)




class Controller():
    """
    This is the core of the data-to-GUI link.
    The main method is respond where it receives signals from the GUI, and performs a task.
    Some attributes are espeically worth mentioning:
    -data: the DataSet object linked to the tracking
    -frame_num: total number of frames
    -channel_num: number of image channels
    -n_neurons: number of neurons
    -pointdat: ground truth annotations
    -NN_pointdat: neural network predictions
    -updated_points: changes from streamed dataset
    """
    def __init__(self,dataset: DataSet,settings,i_init=0):
        """
        Initializes the tracking setup.
        dataset: a DataSet class
        settings: a settings dictionary
        i_init: time slice to initialize
        """
        #this sets up the environment
        self.data = dataset

        self.settings=settings
        logger.info(f"Loading dataset: {self.data.name}")

        self.ready = False

        self.timer = misc.UpdateTimer(1. / int(self.settings["fps"]), self.update)

        # whether data is going to be as points or as masks:
        
        self.point_data = self.data['point_data'][0] if self.data['point_data'] else None

        self.frame_num = self.data.frame_num
        self.data_name = self.data.name

        #we fetch the useful information of the dataset, the h5 file is initialized here.
        self.channel_num = self.data.nb_channels
        self.n_neurons = self.data.nb_neurons   # Todo: Warning, this can change with masks but not with points
        self.frame_shape = self.data.frame_shape      # Todo: make sure that changes in shape due to cropping do not matter

        self.NNmask_key=""

        # all points are loaded in memory
        self.pointdat = self.data.pointdat
        if self.point_data:
            self.pointdat = self.data.pointdat
        else:   # either masks, or yet unknown
            self.pointdat = np.full((self.frame_num,self.n_neurons + 1, 3), np.nan)
        self.neuron_presence = self.data.neuron_presence   # self.frame_num * self.n_neurons+1 array of booleans
        # todo Warning, this will be saved automatically in mask mode but not in point mode (in point mode it is saved only when pointdat is saved)

        self.ci_int = np.zeros((self.n_neurons, self.frame_num, 2))  
        if self.neuron_presence is None or self.frame_num > self.neuron_presence.shape[0]:
            self._fill_neuron_presence()
        elif self.frame_num > self.neuron_presence.shape[0]:
            self._fill_neuron_presence()

        self.NN_pointdat = np.full((self.frame_num,self.n_neurons+1,3),np.nan)
        self.NN_or_GT = np.where(np.isnan(self.pointdat),self.NN_pointdat,self.pointdat)   # TODO AD: init using method?

        #this is about the time setting and tracks
        self.i=i_init
        n_row=int(self.settings["tracks_num_row"])
        box_ind=self.i//n_row
        labs=[lab for lab in range(box_ind*n_row,min((box_ind+1)*n_row,self.frame_num))]
        self.curr_labs=labs

        #now these are about the main figure image
        self.log_const=np.log(2)
        self.high=100
        self.low=0
        self.notfound=np.flip(np.load(self.settings["notfound"]),1)


        # SJR: to prevent double updates
        self.justupdated=0

        self.box_details = [1,1,1,0]#MB: for initialization of box specification for annotation in boxing mode

        #these options should match the GUI initializing options of checkboxes
        self.options={}
        self.options["overlay_pts"]=True
        self.options["autosave"]=False
        self.options["overlay_adj"]=False
        self.adj=-1
        self.options["overlay_NN"] = True   # this is for points only
        self.options["overlay_mask"]=int(self.settings["overlay_mask_by_default"])
        self.mask_thres=int(self.settings["mask_threshold_for_new_region"])
        self.options["overlay_act"]=True
        self.options["mask_annotation_mode"] = False
        self.options["RenumberComp"] = False#MB added to be able to renumber only one component of a cell
        self.options["defining_cropzone_mode"] = False   # AD Whether we are in the process of defining a cropping zone
        self.options["follow_high"]=True
        self.options["overlay_tracks"]=True
        self.options["first_channel_only"]=int(self.settings["just_show_first_channel"])
        self.options["second_channel_only"]=False
        self.options["save_crop_rotate"] = False
        self.options["save_blurred"] = False
        self.options["save_subtracted_bg"] = False
        self.options["save_1st_channel"] = False
        self.options["save_green_channel"] = False
        self.options["save_resized"] =  False
        self.options["AutoDelete"] = False
        self.options["save_after_reversing"] = False
        self.options["save_after_reversing_Cuts"] = False
        self.options["generate_deformation"] = False
        self.options["use_old_trainset"] = False
        self.options["boxing_mode"] = False
        self.options["ShowDim"] = True


        self.tr_fut=5
        self.tr_pst=-5
        self.options["autocenter"] = False
        self.autocenter_peakmode=True
        self.autocenterxy=3
        self.autocenterz=2
        self.autocenter_kernel=np.array(np.meshgrid(np.arange(-self.autocenterxy,self.autocenterxy+1),np.arange(-self.autocenterxy,self.autocenterxy+1),np.arange(-self.autocenterz,self.autocenterz+1),indexing="ij")).reshape(3,-1)
        self.peak_thres=4
        self.peak_sep=2
        #SJR: options for blurring the image for viewing and annotation
        self.blur_s=1
        self.blur_b=25
        self.blur_image=False

        self.highlighted=0

        self.button_keys = {}

        self.assigned_sorted_list = []  # AD sorted list of neurons (from1) that have a key assigned
        self.mask_temp=None

        if not self.point_data:
            # get all existing NNs
            self.NNmodels = []
            self.NNinstances = {}
            self._scan_NN_models()
            self._scan_NN_instances()

        self.subprocmanager=SubProcManager.SubProcManager()


        # Here are the lists of clients (typicaly elements of the gui) that have registered to be pinged when some data
        #  changes (typically, to change the display accordingly).
        # here when the time frame changes (just for the time frame; clients that also need other updates such as the
        # image must register to other lists)
        self.frame_registered_clients = []
        # here when the neuron-key assignment changes
        self.neuron_keys_registered_clients = []
        # here when the number of neurons changes
        self.nb_neuron_registered_clients = []
        # here when the neurons present in a frame change
        self.present_neurons_registered_clients = []
        # here when the neurons present in any frame change
        self.present_neurons_all_times_registered_clients = []
        # here when the highlighted neuron changes
        self.highlighted_neuron_registered_clients = []
        # here when the frame content changes
        self.frame_img_registered_clients = []
        # here when the mask for the current frame changes
        self.mask_registered_clients = []
        # here when any of the pointdat changes
        self.points_registered_clients = []
        # here when the links between points changes
        self.pointlinks_registered_clients = []
        # here when the highlighted track data changes
        self.highlighted_track_registered_clients = []
        # here when the z-viewing should change:
        self.zslice_registered_clients = []
        # by wheeling the mouse in MainFigWidget (but that's not hard to change, see methods of MainFigWidget)
        # here when the mask annotation threshold changes
        self.mask_thres_registered_clients = []
        # here when the set of all (finished) NN instances changes
        self.NN_instances_registered_clients = []
        # here when the validation set for current NN instance changes
        self.validation_set_registered_clients = []
        # here when the autocenter mode changes
        self.autocenter_registered_clients = []
        # here when some calcium intensity changes
        self.calcium_registered_clients = []
        # here when the gui is disabled during NN run
        self.freeze_registered_clients = [self.timer]

        # list of things that change when the time frame changes:
        # time_frame, frame_img, present_neurons, mask, all kinds of pointdats,

        # EPFL lab specific processes
        self.segmenter = None
        self.ref_frames = set()   # the set of frames that the user wants to use for the next registration
        GlobalParameters.set_params()
        self.color_manager = misc.ColorAssignment(self)

        # calcium activities
        #TODO set self ci_int

        self.updated_points = {}
        
        
    def set_point_data(self, value:bool):
        self.data.set_point_data()
        self.point_data = value
        self.pointdat = self.data.pointdat

    def set_up(self):
        # now we actually initialize
        #import pdb; pdb.set_trace()
        self.select_frames()
        self.ready = True
        self.signal_nb_neurons_changed()
        self.update(t_change=True)

    def _fill_neuron_presence(self, frame_num_change=False):
        if frame_num_change:
            old_pres = self.neuron_presence
            old_t = old_pres.shape[0]
            self.neuron_presence = np.full((self.frame_num, self.n_neurons + 1), False)
            self.neuron_presence[:old_t] = old_pres
        else:
            old_t = 0
            self.neuron_presence = np.full((self.frame_num, self.n_neurons + 1), False)
        if self.point_data:
            self.neuron_presence = ~np.isnan(self.pointdat[:, :, 0])
        elif self.point_data is None:
            pass
        else:
            for t in range(old_t, self.frame_num):
                mask = self.data.get_mask(t)
                if mask is not False:
                    present = np.unique(mask)
                    if present[0] == 0:   # should be the case (background)
                        present = present[1:]
                    self.neuron_presence[t, present] = True
        
        self.data.neuron_presence = self.neuron_presence

    def update(self, t_change=False):
        """
        Update the controller state, including images and client notifications.
        """
        if not self.ready:  # Prevent updates during initialization
            return

        # Ensure updates happen only if allowed by timer
        if not self.timer.update_allowed(t_change):
            return

        # Save automatically if autosave is enabled
        if self.options["autosave"]:
            self.save_status()

        # Handle time change event
        if t_change:
            for client in self.frame_registered_clients:
                client.change_t(self.i)

            # Clear temporary mask for new frames in annotation mode
            if self.options["mask_annotation_mode"]:
                self.mask_temp = None

        # Load images through the `data` model
        try:
            if self.channel_num == 2:
                if self.options["first_channel_only"]:
                    self.im_rraw = self.data.get_frame(self.i, col="red")
                    self.im_graw = None
                elif self.options["second_channel_only"]:
                    self.im_rraw = self.data.get_frame(self.i, col="green")
                    self.im_graw = None
                else:
                    self.im_rraw = self.data.get_frame(self.i, col="red")
                    self.im_graw = self.data.get_frame(self.i, col="green")
            else:
                self.im_rraw = self.data.get_frame(self.i, col="red")
                self.im_graw = None

            if self.options["ShowDim"]:
                print("Frame dimensions: " + str(np.shape(np.array(self.im_rraw))))
                self.options["ShowDim"] = False

            # Notify clients of updated frame image data
            for client in self.frame_img_registered_clients:
                client.change_img_data(np.array(self.im_rraw), np.array(self.im_graw))
            
            threading.Thread(target=self.data.prefetch_frames, args=(self.i, "red"), daemon=True).start()
        except Exception as e:
            print(f"Error updating frame data: {str(e)}")
        #load the mask, from ground truth or neural network
        #show mask if either the "overlay mask" OR the "mask annotation mode" checkboxes are checked
        self.update_mask_display()

        self.signal_pts_changed(t_change=t_change)

        present_neurons = np.flatnonzero(self.neuron_presence[self.i])

        for client in self.present_neurons_registered_clients:
            client.change_present_neurons(present_neurons)

        #peak calculation is only when requested
        self.peak_calced=False

    def _show_masks(self):
        """
        In mask mode, whether we are currently showing masks.
        :return: bool
        """
        return (self.options["overlay_mask"] or self.options["mask_annotation_mode"]
                or self.options["boxing_mode"] or self.data.only_NN_mask_mode)

    def valid_points_from_all_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            logger.warning("No valid points found. Returning empty array.")
            return np.zeros((0, 3))
        return points[~np.isnan(points).any(axis=1)]


    @property
    def i(self):
        if self._fixed_frame:
            return self._i
        else:
            return self.selected_frames[self._i]

    @i.setter
    def i(self, value):
        self._fixed_frame = True
        self._i = value

    def move_relative_time(self, relative_t: int):
        """
        Move by relative_t time frames forward or backward (among selected_frames) (and update everything)
        :param relative_t: positive (forward) or negative (backward) int
        """

        if self.i not in self.selected_frames:
            # cannot navigate starting from frame that is not in selected_frames, start from closest selected frame
            closest_selected = len(self.selected_frames) - 1
            for i in self.selected_frames:
                if i > self.i:
                    closest_selected = i
                    break
            self.i = closest_selected
            relative_t -= 1   # we have already moved right, so move right by one less
        if self._fixed_frame:
            self._i = self.selected_frames.index(self._i)
            self._fixed_frame = False

        test_i = self._i + relative_t
        if (test_i < 0) or (test_i >= len(self.selected_frames)):
            return
        self._i = test_i

        # Finally, update according to new time
        self.update(t_change=True)

    def go_to_frame(self, t):
        """Moves to frame t and updates everything accordingly"""
        # set the absolute time
        if t < 0 or t >= self.frame_num:
            return
        self.i = t
        self.update(t_change=True)

    def recompute_point_presence(self):
        """
        To be called when self.pointdat or self.NN_pointdat is modified
        """
        self.NN_or_GT = np.where(np.isnan(self.pointdat), self.NN_pointdat, self.pointdat)
        self.neuron_presence = ~np.isnan(self.NN_or_GT[:, :, 0])

    def update_mask_display(self):
        """
        To be called when on of the parameters of the mask display (such as whether to display the mask, or to display
        only the NN mask...) is modified (the mask may be modified too, but then make sure mask_change is called... but
        I don't think we ever modify both at the same time).
        If only the mask itself changes, call mask_change instead.
        """
        if self._show_masks():
            mask = self.data.get_mask(self.i, force_original=False)
            if mask is False or self.data.only_NN_mask_mode:
                mask = self.data.get_NN_mask(self.i, self.NNmask_key)
                if mask is False:
                    self.mask = None
                else:
                    self.mask = mask
            else:
                self.mask = mask
        else:
            self.mask = None
        for client in self.mask_registered_clients:
            client.change_mask_data(self.mask)

    def signal_pts_changed(self, t_change=True):
        """
        Updates self.NN_or_GT to match changes in self.pointdat and self.NN_pointdat,
        then re-sends all types of point data.
        Not all may be necessary to re-send in all cases, but this is still better than before (since only the update
        method existed, and would update not only the points but also the image etc)
        """
        # Todo AD: could simplify if only one point was changed...
        #import pdb; pdb.set_trace()
        if not self.point_data:
            return

        self.recompute_point_presence()

        if self.options["follow_high"] and t_change:
            self.center_on_highlighted()

        if self.options["overlay_tracks"] and self.highlighted != 0:
            for client in self.highlighted_track_registered_clients:
                client.change_track(self.highlighted_track_data(self.highlighted))

        pts_dict = {}

        # ground truth points
        if self.options["overlay_pts"]:
            gt_points = self.pointdat[self.i]
            pts_dict["pts_pointdat"] = self.valid_points_from_all_points(gt_points)
        else:
            pts_dict["pts_pointdat"] = np.zeros((0, 3))

        # adjacent in time points
        if self.options["overlay_adj"] and not ((self.i + self.adj) < 0 or (self.i + self.adj) >= self.frame_num):
            pts_adj = self.NN_or_GT[self.i + self.adj]
            pts_dict["pts_adj"] = self.valid_points_from_all_points(pts_adj)
        else:
            pts_adj = None
            pts_dict["pts_adj"] = np.zeros((0, 3))

        # overlay NN points
        if self.options["overlay_NN"]:
            NN_points = np.where(np.isnan(self.pointdat[self.i]), self.NN_pointdat[self.i], np.nan)
            pts_dict["pts_NN_pointdat"] = self.valid_points_from_all_points(NN_points)
        else:
            pts_dict["pts_NN_pointdat"] = np.zeros((0, 3))

        # overlay activated points
        if self.options["overlay_act"]:
            pts_dict["pts_act"] = self.valid_points_from_all_points(self.NN_or_GT[self.i][self.assigned_sorted_list, :])
        else:
            pts_dict["pts_act"] = np.zeros((0, 3))

        # highlighted point
        pts_dict["pts_high"] = self.valid_points_from_all_points(np.array(self.NN_or_GT[self.i][self.highlighted])[None, :])

        for client in self.points_registered_clients:
            client.change_pointdats(pts_dict)

        # links between points
        if self.options["overlay_pts"] and self.options["overlay_NN"] and pts_adj is not None and not (
                len(pts_adj) != len(gt_points) or len(pts_adj) != self.n_neurons + 1):
            pts = np.where(np.logical_not(np.isnan(gt_points)), gt_points, NN_points)
            valids = np.logical_not(np.isnan(pts[:, 0])) * np.logical_not(np.isnan(pts_adj[:, 0]))
            link_data = np.concatenate([pts[valids], pts_adj[valids]], axis=0)
        else:
            link_data = None
        for client in self.pointlinks_registered_clients:
            client.change_links(link_data)

    def signal_present_all_times_changed(self):
        for client in self.present_neurons_all_times_registered_clients:
            client.change_present_neurons_all_times(self.neuron_presence)

    def signal_nb_neurons_changed(self):
        """
        Method to signal to all registered clients that the neumber of neurons has changed.
        """
        if self.highlighted > self.n_neurons:
            self.highlighted = 0
        for client in self.nb_neuron_registered_clients:
            client.change_nb_neurons(self.n_neurons)

    def mask_change(self, t=None):
        """
        Updates everything that needs to be updated when the mask for the current time frame is modified.
        If added_neuron_from1 or removed_neuron_from1 is given, it must be the only modification of the mask.
        Warning: if t is not the current time, no mask will be saved to the dataset (it is assumed that it was already
        saved, eg by Segmenter)
        """
        if t is None:
            t = self.i
            mask = self.mask
        else:
            mask = self.data.get_mask(t)
            if t == self.i and self._show_masks() and not self.data.only_NN_mask_mode:
                self.mask = mask

        old_present = np.flatnonzero(self.neuron_presence[t])
        assert 0 not in old_present, "0 corresponds to no neuron and should never be set to present"
        new_present = np.unique(mask)
        if new_present[0] == 0:   # should always be the case (otherwise it means there is no background)
            new_present = new_present[1:]
        if len(new_present) != len(old_present) or np.any(new_present != old_present):
            # at least one neuron has appeared/disappeared from this frame
            # First, increase number of neurons if a new neuron exists
            if len(new_present):
                max_neu = int(max(new_present))
                if max_neu > self.n_neurons:
                    self.n_neurons = max_neu
                    self.data.nb_neurons = self.n_neurons
                    old_presence = self.neuron_presence
                    self.neuron_presence = np.zeros((self.frame_num, self.n_neurons + 1), dtype=bool)
                    self.neuron_presence[:, :old_presence.shape[1]] = old_presence
                    self.signal_nb_neurons_changed()
                elif np.shape(self.neuron_presence)[1] < self.n_neurons:
                    old_presence = self.neuron_presence
                    self.neuron_presence = np.zeros((self.frame_num, self.n_neurons + 1), dtype=bool)
                    self.neuron_presence[:, :old_presence.shape[1]] = old_presence
                    self.signal_nb_neurons_changed()
                # Second, update the presence
                self.neuron_presence[t] = False
                self.neuron_presence[t, new_present] = True
            # Third, reduce number of neurons if neurons have disappeared (from all frames)
            # Todo: that will not reduce the nb of neurons if the last n neurons are absent and already were absent. Is it ok?
            cumsum = np.cumsum(np.sum(self.neuron_presence, axis=0)[::-1])
            if cumsum[0] == 0:   # last neuron is absent at all times
                self.n_neurons = self.n_neurons - np.flatnonzero(cumsum)[0]
                self.data.nb_neurons = self.n_neurons
                self.neuron_presence = self.neuron_presence[:, :self.n_neurons + 1]
                self.signal_nb_neurons_changed()
            # Finally
            if t == self.i:
                for client in self.present_neurons_registered_clients:
                    client.change_present_neurons(present=new_present)
            else:
                self.signal_present_all_times_changed()   # Todo actually only one time changes; and this will be called many times after segmentation
            self.data.neuron_presence = self.neuron_presence

        if t == self.i:
            self.data.save_mask(t, mask, False)
            for client in self.mask_registered_clients:
                client.change_mask_data(self.mask)

        # recompute corresponding calcium activities
        for client in self.calcium_registered_clients:
            client.change_ca_activity(self.ci_int)

    def frame_clicked(self, button, coords):
        # a click with coordinates is received
        if self.options["defining_cropzone_mode"]:
            self.crop_points.append(coords[:2])
        elif self.options["mask_annotation_mode"]:

            """
            SJR: add new neuron / object
            """
            coord = np.round(coords).astype(np.int16)
            # SJR: User pressed the middle button (wheel) = 4, which sets a thresholdim_rraw
            if button == 4:
                # SJR: find the coordinates from the click
                # SJR: I don't know why it is necessary to make sure click is in the image but won't test this
                if 0 <= coord[0] < self.frame_shape[0] and 0 <= coord[1] < self.frame_shape[1] and 0 <= coord[2] < self.frame_shape[2]:
                    # SJR: get threshold
                    new_mask_thres = self.im_rraw[coord[0], coord[1], coord[2]]
                    self.set_mask_annotation_threshold(new_mask_thres)
                print("Set threshold:", self.mask_thres, "from coordinates:", coord[0], coord[1], coord[2])

            # SJR: User right clicks = 2 on the seed point to start creating a fill which will become the new cell
            elif button == 2 and not self.options["RenumberComp"]:
                # creates a dialog window
                dlg = ecv.CustomDialog()#MB removed the argument of the function

                # if the user presses 'ok' in the dialog window it executes the code
                # else it does nothing
                # it also tests that the user has entered some value, that it is not
                # empty and that it is equal or bigger to 0.
                if dlg.exec_() and dlg.entry1.text() != '' and int(dlg.entry1.text()) >= 0:

                    # reads the new value to set and converts it from str to int
                    value = int(dlg.entry1.text())

                    if not (coord[0] >= 0 and coord[0] < self.frame_shape[0] and coord[1] >= 0 and coord[1] < self.frame_shape[1] and coord[
                        2] >= 0 and coord[2] < self.frame_shape[2]):
                        return
                    if self.mask_thres is None:
                        QtHelpers.ErrorMessage("Please enter a correct threshold value.")
                        return
                    # SJR: presumably, this is where the pixels with values above the threshold will be selected
                    regs = sim.label(self.im_rraw >= self.mask_thres)  # SJR: Not sure what regs is
                    ind = regs[0][coord[0], coord[1], coord[2]]
                    if ind == 0:
                        return
                    loc = (regs[0] == ind)

                    if self.mask is None:
                        self.mask = np.zeros(self.frame_shape)
                    self.mask_temp = self.mask.copy()

                    self.mask[loc] = value  # SJR: used to have on the RHS: self.highlighted
                    self.mask_change()

            #MB added this extra condition to renumber only one connected component of the mask, not the whole object
            #this function works in either of mask annotation mode and boxing mode
            elif button == 2 and self.options["RenumberComp"]:
                if not (coord[0] >= 0 and coord[0] < self.frame_shape[0] and coord[1] >= 0 and coord[1] < self.frame_shape[1] and coord[
                    2] >= 0 and coord[2] < self.frame_shape[2]):
                    return

                regs = sim.label(self.mask==self.highlighted)
                ind=regs[0][coord[0],coord[1],coord[2]]#MB: label assigned to the coordinate of the clicked point
                if ind==0:
                    return
                loc=(regs[0]==ind)#MB: location of the all the pixels connected to the clicked point(all with the same label)

                self.mask_temp= self.mask.copy()

                self.mask[loc] = self.options["RenumberComp"]
                self.mask_change()

                self.options["RenumberComp"] = False

            # SJR: User left clicks (arg[0]==1) to select which cell to delete
            elif button == 1 and not self.options["boxing_mode"]:
                if self.mask is None:
                    print("Cannot delete a cell from a non-existing or not shown mask.")
                    return
                if not (0 <= coord[0] < self.frame_shape[0] and 0 <= coord[1] < self.frame_shape[1] and 0 <= coord[2] < self.frame_shape[2]):
                    return
                sel = self.mask[coord[0], coord[1], coord[2]]
                if sel != 0:   # highlight/unhighlight the clicked neuonr
                    self.highlight_neuron(sel)
                else:   # unhighlight all
                    self.highlight_neuron(self.highlighted)
        #MB: to ba able to draw boxes around objects of interest
        if self.options["boxing_mode"]:   # Todo make cases clearer (esp for points)
            if self.box_details is None:
                QtHelpers.ErrorMessage("Please enter the box details in the correct format.")
                return

            w,h,d,box_id = self.box_details
            coord = np.array(coords).astype(np.int16)
            if button == 1 and not self.options["RenumberComp"]:#left clicks are only accepted
                if not (coord[0] >= 0 and coord[0] < self.frame_shape[0] and coord[1] >= 0 and coord[1] < self.frame_shape[1] and coord[
                    2] >= 0 and coord[2] < self.frame_shape[2]):
                    return
                self.mask_temp= self.mask.copy()#save to allow undo

                self.mask[coord[0]:coord[0]+w,coord[1]:coord[1]+h,coord[2]:coord[2]+d] = box_id
                self.mask_change()

            if button == 2 and self.options["RenumberComp"]:
                if not (coord[0] >= 0 and coord[0] < self.frame_shape[0] and coord[1] >= 0 and coord[1] < self.frame_shape[1] and coord[
                    2] >= 0 and coord[2] < self.frame_shape[2]):
                    return

                regs = sim.label(self.mask==self.highlighted)
                ind=regs[0][coord[0],coord[1],coord[2]]#MB: label assigned to the coordinate of the clicked point
                if ind==0:
                    return
                loc=(regs[0]==ind)#MB: location of the all the pixels connected to the clicked point(all with the same label)

                self.mask_temp= self.mask.copy()

                self.mask[loc] = self.options["RenumberComp"]
                self.mask_change()

                self.options["RenumberComp"] = False

        elif None not in coords:  # skip when the coordinate is NaN
            if not self.point_data:
                return
            # assert self.point_data, "Not available with mask data."   # Todo could be implemented
            coord = np.array(coords).astype(np.int16)
            # this will click on the nearest existing annotation
            existing_annotations = np.logical_not(np.isnan(self.NN_or_GT[self.i][:,0]))
            indarr = np.nonzero(existing_annotations)[0]
            if len(indarr) == 0:
                return
            tree_temp = spat.cKDTree(self.NN_or_GT[self.i][indarr, :3])
            d, ii = tree_temp.query(coord)
            i_from1 = indarr[ii]
            self.highlight_neuron(i_from1)

    def key_pressed(self, key, coords):
        # user hits keyboard. annotates or deletes neuron
        if (None in coords) or (np.isnan(np.sum(coords))):
            return
        if not self.point_data:
            QtHelpers.ErrorMessage("Pressing a key is not available for masks.")   # Todo: could be implemented?
            return
        coords = np.array(coords)
        if key == "d":
            indarr = np.where(np.logical_not(np.isnan(self.NN_or_GT[self.i][:, 0])))[0]
            if len(indarr) == 0:
                return
            tree_temp = spat.cKDTree(self.NN_or_GT[self.i][indarr, :3])
            d, ii = tree_temp.query(coords)
            i_from1 = indarr[ii]
            if d < 5:
                self.registerpointdat(i_from1, None, rm=True)
                self.NN_pointdat[self.i, i_from1] = np.nan
                self.update()
            return
        if key.lower() not in self.button_keys.keys():
            return
        if self.options["autocenter"] and not key.isupper():
            coords = self.do_autocenter(coords)
        self.registerpointdat(self.button_keys[key.lower()], coords)

    def rotate_frame(self, angle: float):
        """Rotates current frame around its center and replaces frame by rotated version in data"""
        if self.channel_num != 2:
            print("Only for Hlab")
            return

        im_red = self.data.get_frame(self.i).astype(float) / 255
        im_green = self.data.get_frame(self.i, col="green").astype(float) / 255
        rot_mat = cv2.getRotationMatrix2D((im_red.shape[1] / 2, im_red.shape[0] / 2), angle, 1.0)
        imr = cv2.warpAffine(im_red, rot_mat, (im_red.shape[1], im_red.shape[0]))
        img = cv2.warpAffine(im_green, rot_mat, (im_green.shape[1], im_green.shape[0]))
        imrot_red = np.clip(imr * 255, 0, 255).astype(np.int16)
        imrot_green = np.clip(img * 255, 0, 255).astype(np.int16)

        self.data.replace_frame(self.i, imrot_red, imrot_green)
        print("Frame", self.i, "rotated")
        self.update()

    def define_crop_region(self):
        """
        Toggles self.options["defining_cropzone_mode"]
        If leaving the cropzone definition mode, computes the desired crop zone coordinates from saved points, and saves
        this crop zone to self.data.
        """
        self.options["defining_cropzone_mode"] = not self.options["defining_cropzone_mode"]
        if self.options["defining_cropzone_mode"]:
            # enters cropzone definition mode, creates an empty list of points.
            # These points are meant to delimit the boundaries of the region to crop. More precisely, the cropped region
            # will be the tightest possible rectangle that includes all these points.
            self.crop_points = []
        else:
            # leaves cropzone definition mode and computes and saves the crop zone
            points = np.array(self.crop_points)
            xleft, yleft = points.min(axis=0)
            xright, yright = points.max(axis=0)
            xleft = int(xleft)
            xright = int(np.ceil(xright))
            yleft = int(yleft)
            yright = int(np.ceil(yright))
            self.data.save_ROI_params(xleft, xright, yleft, yright)
            del self.crop_points

    def toggle_old_trainset(self):
        self.options["use_old_trainset"] = not self.options["use_old_trainset"]
        self.update()

    def toggle_add_deformation(self):
        self.options["generate_deformation"] = not self.options["generate_deformation"]
        self.update()

    def toggle_reverse_transform(self):
        self.options["save_after_reversing"] = not self.options["save_after_reversing"]
        self.update()

    def toggle_undo_cuts(self):
        self.options["save_after_reversing_Cuts"] = not self.options["save_after_reversing_Cuts"]
        self.update()

    def toggle_save_crop_rotate(self):
        self.options["save_crop_rotate"] = not self.options["save_crop_rotate"]
        self.update()

    def toggle_save_subtracted_bg(self):
        self.options["save_subtracted_bg"] = not self.options["save_subtracted_bg"]
        self.update()

    def toggle_save_1st_channel(self):
        self.options["save_1st_channel"] = not self.options["save_1st_channel"]
        self.update()

    def toggle_save_green_channel(self):
        self.options["save_green_channel"] = not self.options["save_green_channel"]
        self.update()

    def toggle_save_blurred(self):
        self.options["save_blurred"] = not self.options["save_blurred"]
        self.update()

    def toggle_save_resized_img(self):
        self.options["save_resized"] = not self.options["save_resized"]
        self.update()

    def toggle_auto_delete(self):
        self.options["AutoDelete"] = not self.options["AutoDelete"]
        self.update()

    def import_mask_from_external_file(self,Address,transformation_mode,green=False):
        """
        MB defined this to import from another file that contains another NN run
        on our current file or th ecropped and rotated version of the file.
        """
        self.save_status()
        self.update()
        ExtFile = DataSet.load_dataset(Address)
        frameCheck = self.data.get_frame(0, col= "red")#We compare this with dimensions of the uploaded masks
        shapeCheck = np.shape(frameCheck)
        transformBack = self.options["save_after_reversing"]
        UndoCuts = self.options["save_after_reversing_Cuts"]
        for t in range(ExtFile.dataset.attrs["T"]):
            #Assuming the imported file was a derivative of the current file after cropping, rotating and other preprocesses,
            #fr t of imported file corresponds to frame "orig_index" of the current file
            origIndex = ExtFile.get_frame_match(t)   # original number of frame that t corresponds to is saved in this dataset
            if origIndex is False:
                print("The map between the two files frame numbers is not found. It is taken to be identity")
                origIndex = t

            maskTemp = ExtFile.get_mask(t, force_original=True)   # don't apply crop and rotate on the imported masks
            if maskTemp is not False and transformBack and origIndex < self.frame_num:
                ExtFile.crop = True
                ExtFile.align = True

                img = maskTemp

                if transformation_mode:
                    origTrans,offset = ExtFile.get_transfoAngle(origIndex)
                else:
                    origTrans = self.data.get_transformation(origIndex)
                origROI = self.data.get_ROI_params()
                #copy the cropping parameters of the imported file temporarily into the current file:
                if origROI is not None and origTrans is not None:
                    TemptransParam = 1
                else:
                    TemptransParam = 0
                self.data.save_ROI_params(*ExtFile.get_ROI_params())

                if transformation_mode==0:
                    centerRot=0
                    self.data.save_transformation_matrix(origIndex, ExtFile.get_transformation(t))
                else:
                    centerRot=1
                    mat = np.zeros(4)
                    mat[0],mat[1:] = ExtFile.get_transfoAngle(t)
                    self.data.save_transformation_matrix(origIndex,mat,1)
                OrigCrop = self.data.crop
                OrigAlign =self.data.align
                self.data.crop = True   # so that the save_mask function applies the reverse transformations
                self.data.align = True

                if UndoCuts:
                    Zvec = ExtFile.original_intervals("z")
                    Yvec = ExtFile.original_intervals("y")
                    Xvec = ExtFile.original_intervals("x")
                    imgT = np.zeros((shapeCheck[0],shapeCheck[1],shapeCheck[2]))
                    imgT[int(Xvec[0]):int(Xvec[1]),int(Yvec[0]):int(Yvec[1]),int(Zvec[0]):int(Zvec[1])] = img#np.shape(img)[2]] = img
                    if green:
                        self.data.save_green_mask(origIndex, imgT, False,centerRot)
                    else:
                        self.data.save_mask(origIndex, imgT, False,centerRot)

                elif not np.shape(img)[2] == shapeCheck[2]:
                    print("Z dimensions doesn't match. Zero entries are added to mask for compensation")
                    Zvec = ExtFile.original_intervals("z")
                    print(Zvec)
                    Zvec = [0,31]
                    imgT = np.zeros((np.shape(img)[0],np.shape(img)[1],shapeCheck[2]))
                    imgT[:,:,int(Zvec[0]):int(Zvec[1])] = img[:,:,int(Zvec[0]):int(Zvec[1])]
                    if green:
                        self.data.save_green_mask(origIndex, imgT, False,centerRot)
                    else:
                        self.data.save_mask(origIndex, imgT, False, centerRot)
                else:
                    if green:
                        self.data.save_green_mask(origIndex, img, False,centerRot)
                    else:
                        self.data.save_mask(origIndex, img, False,centerRot)
                if TemptransParam == 1:   # Todo I think this just saves what was loaded identically
                    if transformation_mode==0:
                        self.data.save_transformation_matrix(origIndex, origTrans)
                    else:
                        mat = np.zeros(4)
                        mat[0],mat[1:] = ExtFile.get_transfoAngle(t)
                        self.data.save_transformation_matrix(origIndex,mat,1)

                    self.data.save_ROI_params(*origROI)
                self.data.align = OrigAlign
                self.data.crop = OrigCrop
                self.mask_change(origIndex)
            elif maskTemp is not False and origIndex < self.frame_num:
                img = maskTemp

                if transformation_mode==1:
                    centerRot=1
                else:
                    centerRot=0

                if UndoCuts:
                    Zvec = ExtFile.original_intervals("z")
                    Yvec = ExtFile.original_intervals("y")
                    Xvec = ExtFile.original_intervals("x")
                    imgT = np.zeros((shapeCheck[0],shapeCheck[1],shapeCheck[2]))
                    imgT[int(Xvec[0]):int(Xvec[1]),int(Yvec[0]):int(Yvec[1]),int(Zvec[0]):int(Zvec[1])] = img#np.shape(img)[2]] = img
                    if green:
                        self.data.save_green_mask(origIndex, imgT, False,centerRot)
                    else:
                        self.data.save_mask(origIndex, imgT, False,centerRot)
                elif not np.shape(img)[2] == shapeCheck[2]:
                    print("Z dimensions doesn't match. Zero entries are added to mask for compensation")
                    Zvec = ExtFile.original_intervals("z")
                    if Zvec is None:
                        Zvec = [0,32]
                    imgT = np.zeros((np.shape(img)[0],np.shape(img)[1],shapeCheck[2]))
                    imgT[:,:,int(Zvec[0]):int(Zvec[1])] = img
                    if green:
                        self.data.save_green_mask(origIndex, imgT, True,centerRot)
                    else:
                        self.data.save_mask(origIndex, imgT, True,centerRot)

                else:
                    if green:
                        self.data.save_green_mask(origIndex, img, True,centerRot)
                    else:
                        self.data.save_mask(origIndex, img, True,centerRot)
                self.mask_change(origIndex)
        ExtFile.close()
        print("mask upload finished")

    def Preprocess_and_save(self,frame_int,frame_deleted,Z_interval,X_interval,Y_interval,bg_blur,sd_blur,bg_subt,width,height):
        """
        MB defined this to select and delete the desired frames from the
        original movie or blur, subtract background andcrop in z direction.
        it saves the result in a new .h5 file in the directory of the original input files
        """
        if all(i in frame_deleted or i not in self.selected_frames for i in frame_int):
            print("No frames for new dataset, not doing anything. Please select frames.")
            return
        self.save_status()
        self.update()

        frameCheck = self.data.get_frame(0, col="red")
        fr_shape = np.shape(frameCheck)

        z_0 = int(Z_interval[0])
        z_1 = int(Z_interval[1])

        padXL = 0
        padXR = 0
        padYbottom = 0
        padYtop = 0
        padZlow = 0
        padZhigh = 0

        ## Determine the padding pixels needed in direction of x
        if int(X_interval[0])<0:
            x_0 = 0
            padXL = 0-int(X_interval[0])
        else:
            x_0 = int(X_interval[0])

        if int(X_interval[1])==0:
            x_1 = fr_shape[0]
        elif int(X_interval[1])>fr_shape[0]:
            x_1 = fr_shape[0]
            padXR = int(X_interval[1])-fr_shape[0]
        else:
            x_1 = int(X_interval[1])

        ## Determine the padding pixels needed in direction of Y
        if int(Y_interval[0]) <0:
            y_0 = 0
            padYbottom = 0-int(Y_interval[0])
        else:
            y_0 = int(Y_interval[0])

        if int(Y_interval[1])==0:#doesn't change x and y coordinate if the upper bound is zero
            y_1 = fr_shape[1]
        elif int(Y_interval[1])>fr_shape[1]:
            y_1 = fr_shape[1]
            padYtop = int(Y_interval[1])-fr_shape[1]
        else:
            y_1 = int(Y_interval[1])

        ## Determine the padding pixels needed in direction of Z
        if int(Z_interval[0]) <0:
            z_0 = 0
            padZlow = 0-int(Z_interval[0])
        else:
            z_0 = int(Z_interval[0])

        if int(Z_interval[1])==0:#doesn't change z coordinate if the upper bound is zero
            z_1 = fr_shape[2]
        elif int(Z_interval[1])>fr_shape[2]:
            z_1 = fr_shape[2]
            padZhigh = int(Z_interval[1])-fr_shape[2]
        else:
            z_1 = int(Z_interval[1])


        dset_path=self.data.path_from_GUI
        name = self.data_name
        dset_path_rev = dset_path[::-1]
        key=name+"_CroppedandRotated"
        if '/' in dset_path_rev:
            SlashInd = dset_path_rev.index('/')
            dset_path_cropped = dset_path[0:len(dset_path)-SlashInd]
            newpath = os.path.join(dset_path_cropped,key+".h5")
        else:
            newpath = key+".h5"
        hNew = DataSet.create_dataset(newpath)
        hNew.copy_properties(self.data, except_frame_num=True)
        OrigCrop = self.data.crop
        OrigAlign =self.data.align
        if self.options["save_crop_rotate"]:
            self.data.crop = True
            self.data.align = True
        l=0

        key="distmat"
        if key in self.data.dataset.keys():
            distmat = self.data.dataset[key]   # TODO
            ds = hNew.dataset.create_dataset(key,shape=np.shape(distmat),dtype="f4")
            ds[...]=distmat
            print("distmat saved")
        #change the point annotation in accordance with resized dimensions.
        if 'pointdat' in self.data.dataset.keys() or 'pointdat_old' in self.data.dataset.keys():
            import copy
            if 'pointdat' in self.data.dataset.keys():
                pointkey0 = 'pointdat'
            else:
                pointkey0 = 'pointdat_old'
            S0 =  np.array(self.data.dataset[pointkey0])
            S = copy.deepcopy(S0)
            x00,y00,z00 =np.nonzero(~np.isnan(S0))
            for p in range(len(x00)):
                S[x00[p],y00[p],0]=(S0[x00[p],y00[p],0]-x_0)
                S[x00[p],y00[p],1]=(S0[x00[p],y00[p],1]-y_0)
                S[x00[p],y00[p],2]=(S0[x00[p],y00[p],2]-z_0)
            print("point coordinate aligned")
            S_f = copy.deepcopy(S)
            if self.options["save_resized"]:
                x00,y00,z00 =np.nonzero(~np.isnan(S))
                for p in range(len(x00)):
                    S_f[x00[p],y00[p],0]=(S[x00[p],y00[p],0] * width/fr_shape[0])
                    S_f[x00[p],y00[p],1]=(S[x00[p],y00[p],1] * height/fr_shape[1])
                print("resized points saved")
            hNew.dataset.create_dataset(pointkey0,data = S_f)

        for i in frame_int:
            if i not in frame_deleted and i in self.selected_frames:
                if self.options["AutoDelete"]:
                    kcoarse=str(i)+"/coarse_mask"
                    if kcoarse in self.data.dataset.keys():
                        if len(np.unique(self.data.dataset[kcoarse]))<3:
                            continue

                if hNew.nb_channels==2:
                    if self.options["save_1st_channel"] ^ self.options["save_green_channel"]:
                        frameGr = 0
                    else:
                        frameGr = self.data.get_frame(i, col= "green")
                        frameGr = frameGr[:x_1,:y_1,:z_1]
                        frameGr = frameGr[x_0:,y_0:,z_0:]
                        frmean = int(np.mean(frameGr,axis=(0, 1,2)))
                        frameGr = np.pad(frameGr, ((padXL, padXR),(padYtop, padYbottom), (padZlow,padZhigh)),
                                    'constant', constant_values=((frmean, frmean),(frmean,frmean), (frmean,frmean)))#((frmean, frmean),(frmean,frmean), (frmean,frmean)))
                        if self.options["save_resized"]:
                            frameGr = resize_frame(frameGr,width,height)
                else:
                    frameGr = 0

                if self.options["save_green_channel"] and not self.options["save_1st_channel"]:
                    frameRd = self.data.get_frame(i, col= "green")
                else:
                    frameRd = self.data.get_frame(i, col="red")
                if self.options["save_blurred"]:
                    frameRd = blur(frameRd, bg_blur, sd_blur, self.options["save_subtracted_bg"], bg_subt)
                elif self.options["save_subtracted_bg"]:
                    frameRd = blacken_background(frameRd, bg_subt)
                frameRd = frameRd[:x_1,:y_1,:z_1]
                frameRd = frameRd[x_0:,y_0:,z_0:]
                frmean = int(np.mean(frameRd,axis=(0, 1,2)))
                frameRd = np.pad(frameRd, ((padXL, padXR),(padYtop, padYbottom),  (padZlow,padZhigh)),
                            'constant', constant_values=((frmean, frmean),(frmean,frmean), (frmean,frmean)))#, constant_values=((frameRd, frameRd),(frameRd,frameRd), (frameRd,frameRd)))
                if self.options["save_resized"]:
                    frameRd = resize_frame(frameRd,width,height)

                maskTemp = self.data.get_mask(i)
                if maskTemp is not False:
                    maskTemp = maskTemp[:x_1,:y_1,:z_1]
                    maskTemp = maskTemp[x_0:,y_0:,z_0:]
                    maskTemp = np.pad(maskTemp, ( (padXL, padXR),(padYtop, padYbottom), (padZlow,padZhigh)),'constant', constant_values=((0, 0),(0,0), (0,0)))
                    hNew.nb_neurons = max(hNew.nb_neurons, len(np.unique(maskTemp)), np.max(maskTemp))
                    if self.options["save_resized"]:
                        maskTemp = resize_frame(maskTemp,width,height,mask=True)
                    hNew.save_frame(l, frameRd, frameGr, mask = maskTemp , force_original=True)
                else:
                    hNew.save_frame(l,frameRd, frameGr ,force_original=True)
                kcoarse=str(i)+"/coarse_mask"
                if kcoarse in self.data.dataset.keys() and not self.options["save_crop_rotate"]:#TODO : check for compatibility with rotation and cropping modes
                    CoarseSegTemp = self.data.dataset[kcoarse]
                    CoarseSegTemp = CoarseSegTemp[:x_1,:y_1,:z_1]
                    CoarseSegTemp = CoarseSegTemp[x_0:,y_0:,z_0:]
                    CoarseSegTemp = np.pad(CoarseSegTemp, ((padXL, padXR),(padYtop, padYbottom), (padZlow,padZhigh)),'constant', constant_values=((0, 0),(0,0),(0,0)))
                    kcoarsel=str(l)+"/coarse_mask"
                    kcoarseSegl=str(l)+"/coarse_seg"
                    hNew.dataset.create_dataset(kcoarsel, data=CoarseSegTemp.astype(np.int16), dtype="i2", compression="gzip")
                    hNew.dataset.create_dataset(kcoarseSegl, data=CoarseSegTemp.astype(np.int16), dtype="i2", compression="gzip")
                    print(i)
                #save the transformation functions for later retrieval
                matrix = self.data.get_transformation(i)
                if self.options["save_crop_rotate"] or (matrix is not None):
                    print(i)
                    hNew.save_transformation_matrix(l, matrix)
                hNew.save_frame_match(i, l)
                real_time = self.data.get_real_time(i)
                if real_time is not None:
                    hNew.save_real_time(l, real_time)
                l = l+1
        if self.options["save_resized"]:
            hNew.save_original_size(fr_shape)

        X_interval0 = self.data.original_intervals("x")
        Y_interval0 = self.data.original_intervals("y")
        Z_interval0 = self.data.original_intervals("z")
        if X_interval is not None:
            if Y_interval[1]==0:
                Y_interval[1]=Y_interval0[1]
            if Y_interval[0]==0 and not Y_interval0[0]==0:
                Y_interval[0]=Y_interval0[0]
            if X_interval[1]==0:
                X_interval[1]=X_interval0[1]
            if X_interval[0]==0 and not X_interval0[0]==0:
                X_interval[0]=X_interval0[0]
        hNew.save_original_intervals(X_interval, Y_interval, Z_interval)

        assert hNew.frame_num == l
        self.data.align = OrigAlign
        self.data.crop = OrigCrop

        hNew.close()


    def highlight_neuron(self, neuron_id_from1, block_unhighlight=False):
        """
        Changes the highlighted neuron. If neuron_id_from1 was already highlighted, unhiglight it.
        Otherwise, unhighlight the highlighted neuron (if any) and highlight neuron_id_from1.
        :param neuron_id_from1:
        :param block_unhighlight: blocks unhighlighting in case neuron_id_from1 is already highlighted (i.e.
            neuron_id_from1 will always be highlighted after calling highlight_neuron(neuron_id_from1, True))
        """

        if neuron_id_from1 == self.highlighted:
            if self.highlighted == 0 or block_unhighlight:   # self.highlighted == 0 is edge case for instance in renumber_mask_obj if the neuron has been deleted completely
                return
            self.highlighted = 0
            for client in self.highlighted_neuron_registered_clients:
                client.change_highlighted_neuron(unhigh=neuron_id_from1)
            if self.options["overlay_tracks"]:
                for client in self.highlighted_track_registered_clients:
                    client.change_track([[], []])
        else:
            hprev = self.highlighted
            self.highlighted = neuron_id_from1
            if self.point_data:
                high_ptdat = self.valid_points_from_all_points(np.array(self.NN_or_GT[self.i, [self.highlighted]]))
            else:
                high_ptdat = None
            for client in self.highlighted_neuron_registered_clients:
                client.change_highlighted_neuron(high=self.highlighted,
                                                 unhigh=(hprev if hprev else None),
                                                 # high_present=bool(not self.point_data or self.existing_annotations[self.highlighted]),
                                                 # unhigh_present=bool(not self.point_data or self.existing_annotations[hprev] if hprev else False),
                                                 high_key=self._get_neuron_key(self.highlighted),
                                                 high_pointdat=high_ptdat)
            if self.options["follow_high"]:
                self.center_on_highlighted()

            if self.options["overlay_tracks"]:
                for client in self.highlighted_track_registered_clients:
                    client.change_track(self.highlighted_track_data(self.highlighted))

    # this assigns the neuron keys without overlap
    def assign_neuron_key(self, i_from1, key):
        key_changes = []   # list of (ifrom1, newkey) that need to be changed
        # For some of the clients it is important that the "removing" change be before the "adding" change, if the key is modified
        if key in self.button_keys:
            i_from1_prev = self.button_keys.pop(key)
            key_changes.append((i_from1_prev, None))

        if i_from1 in self.button_keys.values():
            key_prev = list(self.button_keys.keys())[list(self.button_keys.values()).index(i_from1)]
            self.button_keys.pop(key_prev)

        if len(self.button_keys) == int(self.settings["max_sim_tracks"]):
            QtHelpers.ErrorMessage("Too many keys set(>" + str(len(self.button_keys)) + "). Update settings if needed")
            return

        self.button_keys[key] = i_from1
        key_changes.append((i_from1, key))
        for client in self.neuron_keys_registered_clients:
            client.change_neuron_keys(key_changes)
        print("Assigning key:", key, "for neuron", i_from1)

        self.assigned_sorted_list = sorted(list(self.button_keys.values()))
        self.signal_pts_changed(t_change=False)   # just needed to display the activated neurons

    def _get_neuron_key(self, neuron_id_from1:int):
        """
        Gets the key assigned to neuron_id_from1.
        done in a non-efficient way, but it's ok because the dict is small.
        :param neuron_id_from1: index of neuron
        :return: key assigned to given neuron, or None if no key assigned.
        """
        for key, neu_id_from1 in self.button_keys.items():
            if neu_id_from1 == neuron_id_from1:
                return key
        return None

    def center_on_highlighted(self):
        """
        Changes the z slice to the highlighted neuron (if possible)
        """
        if self.point_data and self.neuron_presence[self.i, self.highlighted]:
            # set z to the highlighted neuron
            new_z = int(self.NN_or_GT[self.i][self.highlighted, 2] + 0.5)
            self.change_z(new_z)

    def change_z(self, new_z):
        for client in self.zslice_registered_clients:
            client.change_z(new_z)

    def toggle_first_channel_only(self):
        self.options["first_channel_only"] = not self.options["first_channel_only"]
        self.update()

    def toggle_second_channel_only(self):
        self.options["second_channel_only"] = not self.options["second_channel_only"]
        self.update()

    def toggle_pts_overlay(self):
        self.options["overlay_pts"] = not self.options["overlay_pts"]
        self.signal_pts_changed(t_change=False)

    def toggle_NN_overlay(self):
        self.options["overlay_NN"] = not self.options["overlay_NN"]
        self.signal_pts_changed(t_change=False)
        self.update()

    def toggle_track_overlay(self):
        self.options["overlay_tracks"] = not self.options["overlay_tracks"]
        if self.options["overlay_tracks"] and self.highlighted != 0:
            for client in self.highlighted_track_registered_clients:
                client.change_track(self.highlighted_track_data(self.highlighted))
        else:
            for client in self.highlighted_track_registered_clients:
                client.change_track([[], []])

    def toggle_adjacent_overlay(self):
        self.options["overlay_adj"] = not self.options["overlay_adj"]
        self.signal_pts_changed(t_change=False)
        self.update()

    def change_track_past(self, value):
        try:
            int(value)
        except:
            return
        else:
            self.tr_pst = int(value)
            self.signal_pts_changed(t_change=False)

    def change_track_future(self, value):
        try:
            int(value)
        except:
            return
        else:
            self.tr_fut = int(value)
            self.signal_pts_changed(t_change=False)

    def change_adjacent(self, value):
        try:
            int(value)
        except:
            return False
        else:
            if int(value) != 0:
                self.adj = int(value)
                self.signal_pts_changed(t_change=False)
                return True
        return False

    def toggle_mask_overlay(self):
        self.options["overlay_mask"] = not self.options["overlay_mask"]
        self.update_mask_display()

    def toggle_NN_mask_only(self): #MB added to check different NN results
        self.data.only_NN_mask_mode = not self.data.only_NN_mask_mode   # TODO: should it really be in self.data??
        self.update_mask_display()

    def toggle_display_alignment(self):
        self.data.align = not self.data.align
        self.update()

    def toggle_display_cropped(self):
        self.data.crop = not self.data.crop
        self.options["ShowDim"] = True
        self.update()

    def toggle_z_follow_highlighted(self):
        """Toggles the option of following the z dimension of the highlighted neuron"""
        if not self.point_data:
            raise NotImplementedError("This could be done if necessary.")  # TODO easy?
        self.options["follow_high"] = not self.options["follow_high"]
        if self.options["follow_high"]:
            self.center_on_highlighted()

    def toggle_autosave(self):
        """
        Toggles the autosave option: when autosave is on, we save whenever update is called
        Then saves a first time if autosave is now on
        """
        self.options["autosave"] = not self.options["autosave"]
        if self.options["autosave"]:
            self.save_status()

    def close(self,arg,msg):
        ####Dependency
        ok,msg=self.subprocmanager.close(arg,msg)
        ####Self Behavior
        if not ok:
            return False,msg
        if arg=="force":
            msg+="Tracking: Not Saving\n"
            return True,msg
        self.save_status()
        msg+="Tracking: Saved Status\n"
        return True,msg

    def flag_all_selected_as_gt(self):
        """Flag all currently selected frames as ground_truth"""
        self.data.flag_as_gt(self.selected_frames)

    def flag_current_as_gt(self):
        """Flag currently visible frame as ground truth and move to the next"""
        self.data.flag_as_gt([self.i])
        self.move_relative_time(1)

    def use_current_as_ref(self):
        """Current frame will be flagged to be used as reference (for the registration process)"""
        self.ref_frames.add(self.i)

    def select_frames(self, fraction=1., frame = 0.,from_frames="all"):
        """
        Changes the set of currently selected frames to be given fraction of given subset of frames.
        If fraction is 0 or there are no frames answering the criterion from_frames, selected frames will NOT change.
        :param fraction: float with values in (0, 0.5] or 1. If 1, all frames of given subset will be selected.
            Otherwise, the subset will be sampled regularly over time, starting with first frame and with time step 1/fraction.
        :param from_frames: string indicating from what subset frames should be selected.
             Must be one of: ["all", "segmented", "non segmented", "ground truth", "non ground truth"]
        """
        if fraction > 0.5 and fraction != 1:
            # because we sample frames regularly with step 1/fraction
            warnings.warn("Only fractions below 0.5 can be chosen, but you asked {}. Not doing anything.".format(fraction))
            return
        current_frame = self.i
        if from_frames == "all":
            frame_pop = self.data.frames
        elif from_frames == "manual selection":
            frame_pop = [int(fr) for fr in frame]
            print(frame_pop)
        elif from_frames == "segmented":
            frame_pop = self.data.segmented_times()
            print("These are the segmented frames:")#MB added
            print(frame_pop)
        elif from_frames == "non segmented":
            frame_pop = set(self.data.frames) - set(self.data.segmented_times())
            print("These are the non segmented frames:")#MB added
            print(frame_pop)
        elif from_frames == "ground truth":
            frame_pop = self.data.ground_truth_frames()
            print("These are the ground truth frames:")#MB added
            print(frame_pop)
        elif from_frames == "non ground truth":
            #frame_pop = self.data.segmented_non_ground_truth#MB changed non_ground_truth_frames to segmented_non_.... I also added set()
            frame_pop = set(self.data.segmented_times())-set(self.data.ground_truth_frames())
            print("These are the non ground truth frames:")#MB added
            print(frame_pop)
        else:
            raise ValueError("Not a valid button name for frame population")
        if not frame_pop:
            warnings.warn("No such frames. Doing nothing.")
            return

        if from_frames == "manual selection":
            self.selected_frames = frame_pop
        else:
            frame_pop = sorted(frame_pop)
            step = int(1 / fraction)
            self.selected_frames = frame_pop[::step]
        if current_frame in self.selected_frames:
            # can stay on this frame
            self.i = current_frame
        else:
            # move to closest selected frame
            closest_selected = len(self.selected_frames) - 1
            for i in self.selected_frames:
                if i > current_frame:
                    closest_selected = i
                    break
            self.i = closest_selected
            self.update(t_change=True)

    ####################################################################################################################
    # Mask-data processing methods

    def toggle_coarse_seg_mode(self):
        """MB added this to be able to change the state of coarse segment independent of segmented_times. Jan2021"""
        self.data.coarse_seg_mode = not self.data.coarse_seg_mode
        self.update()

    def toggle_use_seg_for_feature(self):
        """MB added this so that one can use also segmentation for extracting features multiple times instead of only once"""
        self.data.use_seg_for_feature = not self.data.use_seg_for_feature
        self.update()

    def test_segmentation_params(self):
        """Segments first frame and displays intermediate results so user can adjust segmentation parameters"""
        if self.segmenter is None:
            self.segmenter = Segmenter(self.data, self.data.seg_params)
        self.segmenter.test_segmentation_parameters(frame=self.i)
        # todo: could just display segmentation in main widget instead of using extra plots from the segmentation cache?

    def segment(self):
        """Performs segmentation of selected frames"""
        assert not self.point_data, "Not available for point data."
        if self.segmenter is None:
            warnings.warn("Segmenting all selected frames," +
                          " you may want to test segmentation parameters on a single frame first")
            self.segmenter = Segmenter(self.data, self.data.seg_params)
        self.segmenter.segment(self.selected_frames)
        for t in self.selected_frames:
            self.mask_change(t)

    def extract_features(self):
        feature_builder = FeatureBuilder(self.data)
        feature_builder.extract_features(self.selected_frames)

    def cluster(self):
        clustering = Clustering(self.data, self.data.cluster_params)
        clustering.find_assignment(self.selected_frames)  # MB changed from data.frames to selected_frames
        for t in self.selected_frames:
            self.mask_change(t)

    def classify(self):
        clf = Classification(self.data)
        print("ground truth frames chosen to classify")
        Ground_TruthSet = set(self.data.ground_truth_frames())  # MB added
        print(Ground_TruthSet)  # MB added
        if Ground_TruthSet.difference(set(self.data.segmented_times())):  # MB added for proper error
            print("These GT frames you chose are not segmented so are not passed to classification function:")
            print(Ground_TruthSet.difference(set(self.data.segmented_times())))
        Ground_TruthSet = Ground_TruthSet.intersection(set(self.data.segmented_times()))
        Ground_TruthSet = list(Ground_TruthSet)
        print("segmented ground truth frames used for classification")
        print(Ground_TruthSet)  # MB added
        clf.prepare(Ground_TruthSet)  # MB changed
        # clf.prepare(self.data.ground_truth_frames())#MB removed
        frames = self.data.segmented_non_ground_truth()
        print("The non-gound-truth frames you are classifying:")  # MB added
        print(set(frames))
        clf.find_assignment(frames)
        for t in frames:
            self.mask_change(t)

    def auto_add_gt_by_registration(self):
        """
        Automatically add to ground truth frames that are "sufficiently" well annotated as measured from registration
        distance to reference frames.
        """
        # Todo: will use preexisting registrations, which may be done frmo coarse seg, is that good?
        if self.ref_frames:
            ref_frames = self.ref_frames
        else:
            ref_frames = None
        # Todo: now, assumes that any preexisting registrations were computed with the right segmentation (i.e. coarse)
        # This might not be true if "registration_for_gt" was used earlier
        reg = Register_Rotate(self.data, ref_frames=ref_frames,
                              frames_to_register=self.data.segmented_non_ground_truth())
        reg.add_good_to_ground_truth()

    def compute_rotation(self):
        if self.ref_frames:
            ref_frames = self.ref_frames
        else:
            ref_frames = None
        was_coarse_seg = self.data.coarse_seg_mode
        self.data.coarse_seg_mode = True
        register = Register_Rotate(self.data, ref_frames=ref_frames, frames_to_register=self.selected_frames)
        self.data.coarse_seg_mode = was_coarse_seg

    # mask annotation mode methods:

    def toggle_box_mode(self): #MB
        self.options["boxing_mode"] = not self.options["boxing_mode"]
        if self.options["boxing_mode"]:
            print("Left click on the lower left corner of the box you like to insert")
        self.update()

    def toggle_mask_annotation_mode(self):
        assert not self.point_data, "Not available for point data."
        self.options["mask_annotation_mode"] = not self.options["mask_annotation_mode"]
        if self.options["mask_annotation_mode"]:
            print("Mask Annotation Mode")
        else:
            self.mask_temp = None
        self.update_mask_display()

    def set_mask_annotation_threshold(self, value):   # SJR
        """Set mask threshold"""
        try:
            self.mask_thres = int(value)
        except ValueError:   # "" for instance, before writing the correct number
            self.mask_thres = None
        else:
            for client in self.mask_thres_registered_clients:   # Todo is this useful?
                client.change_mask_thresh(self.mask_thres)

    def set_box_dimensions(self, info): #MB
        """ To help annotation with boxes. updates the dimensions of the box and the cell id we want to assign to it"""
        try:
            box_info = info.split('-')
            dimensions = box_info[0].split(',')
            box_id = box_info[1]
            self.box_details = [int(dimensions[0]), int(dimensions[1]), int(dimensions[2]), int(box_id)]
        except:
            self.box_details = None

    def renumber_mask_obj(self):
        if self.highlighted == 0:
            return
        #MB added: to get the connected components of the mask
        labelArray,numFtr = sim.label(self.mask==self.highlighted)
        if numFtr>1:
            dlg2 = ecv.CustomDialogSubCell()#ask if you want all components change or only one of them
        #ind=regs[0][coord[0],coord[1],coord[2]]

        dlg = ecv.CustomDialog()#which number to change to

        # if the user presses 'ok' in the dialog window it executes the code
        # else it does nothing
        # it also tests that the user has entered some value, that it is not
        # empty and that it is equal or bigger to 0.
        if dlg.exec_() and dlg.entry1.text() != '' and int(dlg.entry1.text()) >= 0:

            # reads the new value to set and converts it from str to int
            value = int(dlg.entry1.text())

            # SJR: save old mask to allow undo
            self.mask_temp = self.mask.copy()   # Todo this will not work (self.mask=None) if not showing the masks
            if numFtr >1:#MB added
                if not dlg2.exec_():
                    # SJR: erase neuron
                    self.mask[self.mask == self.highlighted] = value
                else:
                    print("Right-click on the component you want to renumber")
                    self.options["RenumberComp"] = value
                    return
            else:
                self.mask[self.mask == self.highlighted] = value
            self.mask_change()
            # unhighlight and turn off neuron_bar button, careful!! does an update, which resets self.mask
            self.highlight_neuron(self.highlighted)

    def renumber_All_mask_instances(self, fro, to):
        if self.highlighted == 0:
            return
        if fro > to:
            print("Invalid times")
            return
        dlg = ecv.CustomDialog()#which number to change to
        if dlg.exec_() and dlg.entry1.text() != '' and int(dlg.entry1.text()) >= 0:
            value = int(dlg.entry1.text())
            print("Renumbering", self.highlighted, "fro", fro, "to", to)
            if not self.point_data:#MB added this to use this feature for epfl data
                for k in range(fro,to):
                    mask_k = self.data.get_mask(k, force_original=False)  # MB added
                    if mask_k is not False:
                        mask_k[mask_k == self.highlighted] = value
                        self.data.save_mask(k, mask_k, False)
                        self.mask_change(k)

                self.highlight_neuron(self.highlighted)

    def permute_masks(self, Permutation):
        self.mask_temp = self.mask.copy()
        for l in range(len(Permutation)-1):
            k = Permutation[l]
            print(k)
            self.mask[self.mask_temp == k] = Permutation[l+1]
            self.mask_change()
            #self.highlight_neuron(self.highlighted)

    def delete_mask_obj(self):
        if self.highlighted == 0:
            return
        # SJR: save old mask to allow undo
        self.mask_temp = self.mask.copy()
        # SJR: erase neuron
        self.mask[self.mask == self.highlighted] = 0
        self.mask_change()
        print("SJR: self.mask.max() in delete_mask_obj", self.mask.max())
        # unhighlight and turn off neuron_bar button, careful!! does an update, which resets self.mask
        self.highlight_neuron(self.highlighted)
        print("SJR: self.mask.max() in delete_mask_obj", self.mask.max())
        print("SJR: self.data.nb_neurons in delete_mask_obj", self.n_neurons)

    def delete_All_mask_instances(self, fro, to):
        if self.highlighted == 0:
            return
        if fro > to:
            print("Invalid times")
            return

        if not self.point_data:#MB added this to use this feature for epfl data
            for k in range(fro,to):
                mask_k = self.data.get_mask(k, force_original=False)  # MB added
                if mask_k is not False:
                    mask_k[mask_k == self.highlighted] = 0
                    self.data.save_mask(k, mask_k, False)
                    self.mask_change(k)

        print("uccessfully deleted "+ str(self.highlighted) + " in all selected frames")
        # unhighlight and turn off neuron_bar button, careful!! does an update, which resets self.mask
        self.highlight_neuron(self.highlighted)

    def undo_mask(self):
        if self.options["mask_annotation_mode"] or self.options["boxing_mode"]:
            if self.mask_temp is not None:
                self.mask = self.mask_temp
                self.data.save_mask(self.i, self.mask)
                self.mask_change()


    # NN related methods

    def clear_frame_NN(self):
        """Deletes all NN predictions in current frame"""
        print("Deleting all annotations in frame", self.i)
        self.pointdat[self.i] = np.nan
        self.NN_pointdat[self.i] = np.nan
        # self.neuron_presence[self.i] = False   # now useless due to recompute_point_presence in update()
        for client in self.calcium_registered_clients:
            client.change_ca_activity(self.ci_int)
        self.update()

    def clear_NN_selective(self, fro, to):
        """
        Deletes the NN predictions within a time range for the highlighted neuron
        :param fro, to: first and last frames for which to delete the NN predictions
        """
        if self.highlighted == 0:
            print("bug cfpark00@gmail.com")
            return
        if fro > to:
            print("Invalid times")
            return
        print("Deleting", self.highlighted, "from", fro, "to", to)
        if self.point_data:
            self.pointdat[fro:to + 1, self.highlighted, :] = np.nan
            self.NN_pointdat[fro:to + 1, self.highlighted, :] = np.nan
            # self.neuron_presence[fro:to + 1, self.highlighted] = False   # now useless due to recompute_point_presence in signal_pts_changed in update

            for client in self.calcium_registered_clients:
                client.change_ca_activity(self.ci_int)
            self.update()
        else:   # MB added this to use this feature for epfl data
            for k in range(fro,to):
                mask_k = self.data.get_mask(k, force_original=False)   # MB added
                if mask_k is not False:
                    mask_k[mask_k == self.highlighted] = 0
                    self.mask_change(k)
            self.highlight_neuron(self.highlighted)   # todo: why not for point_data too?
    def debug_trace(self):
      '''Set a tracepoint in the Python debugger that works with Qt'''
      from PyQt5.QtCore import pyqtRemoveInputHook

      from pdb import set_trace
      pyqtRemoveInputHook()
      set_trace()
    #TODO delete the x thing everywhere 
    def update_ci(self, t, x=None):
      self.data.send_ci_int_patch_to_server(frame=t, settings=self.settings)
      self.ci_int = self.data.ca_act
      self.debug_trace()

    def approve_selective(self, fro, to):
        """
        Approves the NN predictions within a time range for the highlighted neuron.
        I think this is only for pointdat.
        :param fro, to: first and last frames for which to approve the NN predictions
        """
        if self.highlighted == 0:
            print("bug cfpark00@gmail.com")
            return
        if fro > to:
            print("Invalid times")
            return
        print("Approving", self.highlighted, "from", fro, "to", to)
        self.pointdat[fro:to + 1, self.highlighted, :] = np.where(
            np.isnan(self.pointdat[fro:to + 1, self.highlighted, :]), self.NN_pointdat[fro:to + 1, self.highlighted, :],
            self.pointdat[fro:to + 1, self.highlighted, :])

        for client in self.calcium_registered_clients:
            client.change_ca_activity(self.ci_int)
        self.update()

    def select_NN_instance_points(self, helper_name:str):
        """Loads neural network point predictions"""
        self.save_status()
        if not self.point_data:
            return
        if helper_name is None:
            self.NN_pointdat = np.full_like(self.pointdat, np.nan)
            self.update()
            return
        out = self.data.get_method_results(helper_name)
        if out is False:
            self.NN_pointdat = np.full_like(self.pointdat, np.nan)
        else:
            self.NN_pointdat = out
            self.NN_pointdat[:, 0, :] = np.nan
        self.update()
        self.signal_present_all_times_changed()

    def select_NN_instance_masks(self, NetName:str, instance:str):
        """Loads neural network mask predictions"""
        self.save_status()
        if NetName == "":
            self.NNmask_key = ""
            self.update()
            return
        key = NetName + "_" + instance
        self.NNmask_key = key
        validationSet = self.data.get_validation_set(key)   # MB
        for client in self.validation_set_registered_clients:
            client.change_validation_set(validationSet)
        self.update()

    def approve_NN_masks(self):
        """MB added: to set the predicttions of NN for the selected frames as the ground truth"""
        if self.NNmask_key == "":
            print("You should first choose the NN instance")
        else:
            for t in self.selected_frames:
                if True:#not mkey in self.data.dataset.keys():
                    mask = self.data.get_NN_mask(t, self.NNmask_key)
                    if mask is not False:
                        self.data.save_mask(t, mask, False)
                        self.mask_change(t)
                    else:
                        print("There are no predictions for this frame")
    def import_NN(self,Address):
        "save the parameters of NN trained on ExtFile for predicting the masks of the current file"
        ExtFile = DataSet.load_dataset(Address)
        for key in ExtFile.dataset['net']:
            netkey="net/"+key
            self.data.import_external_NN(ExtFile,netkey)
        self.update()
        ExtFile.close()
        print("NN parameters imported successfully")


    def post_process_NN_masks(self, mode, neurons):
        """
        MB added: to post process the predictions of NN for the selected frames as the ground truth
        :param mode: which post-processing to apply
        :param neurons: for modes 1 and 2, neurons to be excluded from post-processing; for modes 3-4-5, neurons to be
            post-processed.
        :return:
        """
        if self.NNmask_key == "":
            print("You should first choose the NN instance")
            return

        def load_fun(t):
            return self.data.get_NN_mask(t, self.NNmask_key)

        def save_fun(t, mask):
            self.data.save_NN_mask(t, self.NNmask_key, mask)

        if mode == 1:
            post_process_NN_masks(self.selected_frames, neurons, load_fun, save_fun)
        elif mode == 2:
            post_process_NN_masks2(self.selected_frames, neurons, load_fun, save_fun)
        elif mode == 3:
            post_process_NN_masks3(self.selected_frames, neurons, load_fun, save_fun)
        elif mode == 4:
            post_process_NN_masks4(self.selected_frames, neurons, load_fun, save_fun)
        elif mode == 5:
            post_process_NN_masks5(self.selected_frames, neurons, load_fun, save_fun)
        self.update()

    def run_NN_masks(self, modelname, instancename, fol, epoch, train, validation, targetframes,pred_mode=False):
        # Todo AD could this be factorized in some way?
        # run a mask prediction neural network
        self.save_status()
        if modelname == "RGN":
            return False, "RGN cannot be used for masks"

        dset_path = self.data.path_from_GUI
        name = self.data.name

        # Check that the number of train/validation frames fits into the available number of frames
        nb_available_frames = len(self.data.segmented_times(force_regular_seg=True))
        if not pred_mode:
            if train + validation > nb_available_frames:
                return False, f"The sum of the number of train and validation frames cannot be greater than the number of annotated frames ({nb_available_frames})"

        # temporary close
        if "_" in name:
            dividedName = name.split("_")
            name = dividedName[0]
        key = name + "_" + modelname + "_" + instancename
        newpath = os.path.join("data", "data_temp", key + ".h5")
        newlogpath = os.path.join("data", "data_temp", key + ".log")
        if key in self.subprocmanager.runnings.keys():
            return False, "This run is already running."
        if os.path.exists(newpath):
            return False, "There is an unpulled instance of this run."

        # we are safe now.
        self.data.close()  # close
        shutil.copyfile(dset_path, newpath)  # whole data set is copied in newpath
        self.data = DataSet.load_dataset(dset_path)
        if pred_mode:
            args = ["python3", "./src/neural_network_scripts/run_NNmasks_f.py", newpath, newlogpath,"2",str(epoch),"0","0",str(train),str(validation)]
        #setting the arguments of NN script.
        # the ordeer of arguments are: path to dataset, path to log file,3:whether or not generate defrmed frames-orgo to prediction mode,
        #4:number of training epochs. 5:whether or not train on previous training set
        #6:whether or not add the deformed frames?
        #7:based on the previous choices: a.number of deformed frames that are added  to the training set_title b.which deformation trick to use. c.training set number
        #8: validation frames number or number of targeet Frames
        #9: the deformation method used
        elif not self.options["use_old_trainset"] and not self.options["generate_deformation"]:
            args = ["python3", "./src/neural_network_scripts/run_NNmasks_f.py", newpath, newlogpath,"0",str(epoch),"0","0",str(train),str(validation)]
        elif not self.options["use_old_trainset"] and self.options["generate_deformation"]:
            args = ["python3", "./src/neural_network_scripts/run_NNmasks_f.py", newpath, newlogpath,"1","1","1","0","3",str(targetframes),"5"]
        elif self.options["use_old_trainset"] and not self.options["generate_deformation"]:
            args = ["python3", "./src/neural_network_scripts/run_NNmasks_f.py", newpath, newlogpath,"0",str(epoch),"1","1","0"]
        else:
            print("you cannot 'select use old deformation' and add deformation at the same time")
        if fol:
            os.mkdir(key)
            dfd = os.path.join("data", "data_temp")
            os.mkdir(os.path.join(key, "data"))
            os.mkdir(os.path.join(key, "runouterr"))
            os.mkdir(os.path.join(key, "data", "data_temp"))
            nnewpath = os.path.join(dfd, key + ".h5")
            nnewlogpath = os.path.join(dfd, key + ".log")
            shutil.move(newpath, os.path.join(key, nnewpath))
            shutil.copyfile(os.path.join("./src/neural_network_scripts/models", modelname + ".py"),
                            os.path.join(key, modelname + ".py"))
            shutil.copyfile("./src/neural_network_scripts/run_NNmasks_f.py", os.path.join(key, "run_NNmasks_f.py"))
            shutil.copyfile("./src/neural_network_scripts/NNtools.py", os.path.join(key, "NNtools.py"))
            with open(os.path.join(key, "run.sh"), "w") as f:
                if not self.options["use_old_trainset"] and not self.options["generate_deformation"]:
                    Totstring = "0 " + str(epoch)+" 0 "+"0 "+str(train)+ " " +str(validation)
                elif not self.options["use_old_trainset"] and self.options["generate_deformation"]:
                    Totstring =  "1 1 1 0 2 "+ str(targetframes)
                elif self.options["use_old_trainset"] and not self.options["generate_deformation"]:
                    Totstring = "0 " + str(epoch)+" 1 1 0"
                f.write("python3 run_NNmasks_f.py" + " " + nnewpath + " " + nnewlogpath + " " +Totstring)
            return True, ""

        return self.subprocmanager.run(key, args, newlogpath)

    def _scan_NN_models(self):
        """
        Looks for existing NNs in files, and populates the list self.NNmodels with them.
        Only used for initialization.
        """
        for file in glob.glob(os.path.join("src", "neural_network_scripts", "models", "*")):
            if not os.path.isfile(file):
                continue
            name = os.path.split(file)[-1].split(".")[0]
            if name not in self.NNmodels:
                self.NNmodels.append(name)
            if name not in self.NNinstances:
                self.NNinstances[name] = []
        pref_dict = {"CZANet": 0, "FastNet3": 1, "VeryFastNet2": 2, "TrackNet": 3, "UNet": 4}

        def our_preference(key):
            if key in pref_dict:
                return pref_dict[key]
            return 10

        self.NNmodels = sorted(sorted(self.NNmodels), key=our_preference)

    def available_method_results(self):
        return self.data.get_available_methods()

    def _scan_NN_instances(self):
        """
        Looks for existing NN instances in self.data, and populates the dict self.NNinstances with them.
        Only used for initialization.
        """
        for key in self.data.available_NNdats():   # Todo AD why for pointdat only? is it just the name?
            NetName, instance = key.split("_")
            if NetName not in self.NNinstances:
                self.NNinstances[NetName] = []
            if instance not in self.NNinstances[NetName]:
                self.NNinstances[NetName].append(instance)

    def pull_NN_res(self, key: str, success: bool):
        """
        Only used for masks.
        Gets the NN results from the file system and saves them to self.data (if success); deletes corresponding NN
        files to leave the file system clear.
        :param key: NN instance identifier (as usual, includes dataname, NN model name, and instance name)
        :param success: whether the NN ran successfully
        :return val: bool, msg: str : True (to signify that pulling or just deleting happened ok), msg (to say whether
            results were pulled or run was unsucessful and files were just deleted).
        """
        # TODO AD I guess this is where available NN instances should be updated (either here, when the results are ready, or earlier when launching the run)
        NetName, runname = key.split("_")[-2:]
        newpath = os.path.join("data", "data_temp", key + ".h5")
        newlogpath = os.path.join("data", "data_temp", key + ".log")
        if success:
            self.data.pull_NN_results(NetName, runname, newpath)
            val, msg = True, "Pull Success"
            if NetName not in self.NNinstances:
                self.NNinstances[NetName] = []
            self.NNinstances[NetName].append(runname)

            for client in self.NN_instances_registered_clients:
                client.change_NN_instances()
        else:
            print("Deleting ", key)
            val, msg = True, "Deleted"
        os.remove(newlogpath)
        os.remove(newpath)
        self.subprocmanager.free(key)
        return val, msg

    ####################################################################################################################

    def set_autocenter(self, size:int, z=False):
        """
        this updates the autocenter kernel
        """
        if z:
            self.autocenterz = size
        else:
            self.autocenterxy = size

        self.autocenter_kernel = np.array(np.meshgrid(
            np.arange(-self.autocenterxy,self.autocenterxy+1),
            np.arange(-self.autocenterxy,self.autocenterxy+1),
            np.arange(-self.autocenterz,self.autocenterz+1),
            indexing="ij")).reshape(3,-1)

    def set_peak_threshold(self, thresh:int):
        self.peak_thres = thresh

    def set_peak_sep(self, val:int):
        self.peak_sep = val

    def toggle_autocenter(self):
        self.options["autocenter"] = not self.options["autocenter"]
        for client in self.autocenter_registered_clients:
            client.change_autocenter_mode(self.options["autocenter"])

    def toggle_autocenter_peakmode(self):
        """Toggles the autocenter peakmode, which chooses autocenter by peak or by maximum intensity"""
        self.autocenter_peakmode = not self.autocenter_peakmode

    #when needed we calculate the peaks
    def calc_curr_peaks(self):
        """
        This calculates the peaks needed for auto centering the neurons
        """
        self.peaks=skfeat.peak_local_max(self.im_rraw,threshold_abs=self.peak_thres,min_distance=self.peak_sep)
        if len(self.peaks)==0:
            self.peaks=np.zeros((0,3))

        self.peaks_tree=spat.cKDTree(self.peaks)
        self.peak_calced=True

    #this wraps the autocenter method coord in gets transformed
    def do_autocenter(self,coord,mode="max"):
        """
        This calculates the peaks needed for auto centering the neurons
        """
        if self.autocenter_peakmode:
            if self.peak_calced:
                sugg=self.peaks[self.peaks_tree.query(coord,k=1)[1]]
            else:
                self.calc_curr_peaks()
                sugg=self.peaks[self.peaks_tree.query(coord,k=1)[1]]
        else:
            intcoord=(coord+0.5).astype(int)
            sh=self.frame_shape
            if (intcoord[0]-self.autocenterxy<0) or (intcoord[0]+self.autocenterxy>=sh[0]):
                return coord
            if (intcoord[1]-self.autocenterxy<0) or (intcoord[1]+self.autocenterxy>=sh[1]):
                return coord
            if (intcoord[2]-self.autocenterz<0) or (intcoord[2]+self.autocenterz>=sh[2]):
                return coord
            pts=intcoord[:,None]+self.autocenter_kernel

            ws = self.im_rraw.swapaxes(0,1)[pts[0],pts[1],pts[2]]   # TODO should it be im_rraw or im_graw or ??

            w=np.sum(ws)
            if w==0:
                return coord
            if mode=="mean":
                assert False, "not intended"
                #return np.sum(ws[None,:]*pts,axis=1)/w
            if mode=="max":
                sugg=pts[:,np.argmax(ws)]
        if np.abs(sugg[0]-coord[0])>3 or np.abs(sugg[0]-coord[0])>3 or np.abs(sugg[0]-coord[0])>2:
            return coord
        else:
            return sugg

    #this saves the status of the current annotations
    def save_status(self):
        # Probably delete because of autosave
        if self.point_data:
            self.save_pointdat()
        self.data.neuron_presence = self.neuron_presence   # todo I think it could be just for points
        # self.data.ca_act = self.ci_int # TODO delete probably
        self.data.save()
        print("Saved")

    #we save the point data
    def save_pointdat(self):
      for frame, updates in self.updated_points.items():
        for neuron, coord in updates.items():
            self.data.send_pointdat_patch_to_server(frame, neuron, coord)
      self.updated_points.clear()

    def save_and_repack(self):
        print("Repacking")
        dset_path = self._close_data()
        h5utils.repack(dset_path)
        print("Repacked Dataset")
        self._open_data(dset_path)

    def pause_for_NN(self):
        dset_path = self._close_data()
        for client in self.freeze_registered_clients:
            client.freeze()
        return dset_path

    def unpause_for_NN(self, dset_path):
        for client in self.freeze_registered_clients:
            client.unfreeze()
        self._open_data(dset_path)

    def _close_data(self):
        self.save_status()
        dset_path = self.data.path_from_GUI
        self.data.close()
        return dset_path

    def _open_data(self, dset_path):
        self.data = DataSet.load_dataset(dset_path)
        if self.point_data:
            self.pointdat = self.data.pointdat
            for client in self.NN_instances_registered_clients:
                client.change_NN_instances()   # Todo Mahsa do we also want this for masks?
        self.update()
        # TODO if the dataset has changed, some things such as the name may have changed, they should be updated (e.g. self.data_name)

    #when a key for a neuron is clicked the point is now annotated. rm is the remove option
    def registerpointdat(self,i_from1,coord,rm=False):
        assert self.point_data, "Not available for mask data."
        print(coord)
        if rm:
            print("Removing neuron",i_from1,"at time",self.i)
            self.pointdat[self.i][i_from1,:]=np.nan
            # self.neuron_presence[self.i, i_from1] = False   # now useless due to recompute_point_presence in signal_pts_changed
        else:
            if any(np.isnan(self.pointdat[self.i][i_from1])):
                add = True
                # self.neuron_presence[self.i, i_from1] = True   # now useless (idem)
            else:
                add = False
            print("Setting neuron",i_from1,"at time",self.i,"to",coord)
            self.pointdat[self.i][i_from1]=coord

        for client in self.calcium_registered_clients:
            if (self.ci_int is not None):
              client.change_ca_activity(self.ci_int[i_from1-1][:, :2], neuron_id_from1=i_from1)
        
        # NEW 1/14/24
        if self.i not in self.updated_points:
          self.updated_points[self.i] = {}
        self.updated_points[self.i][i_from1] = coord if not rm else None
        
        self.signal_pts_changed(t_change=False)
        if rm:
            for client in self.present_neurons_registered_clients:
                client.change_present_neurons(removed=i_from1)
        elif add:
            for client in self.present_neurons_registered_clients:
                client.change_present_neurons(added=i_from1)

    def neuron_ca_activity(self, neuron_id_from1):
        """
        Gets the array of calcium activity for neuron neuron_id_from1
        :param neuron_id_from1: standard neuron id
        :return activity: array of shape nb_frames * 2 with activity[t] is the activity value and error bars of neuron
            neuron_id_from1 at time t
        """
        return self.ci_int[neuron_id_from1-1][:, :2]

    def times_of_presence(self, neuron_idx_from1):
        if self.point_data:
            existing_annotations = np.logical_not(np.isnan(self.NN_or_GT[:, neuron_idx_from1, 0]))
        else:
            existing_annotations = self.neuron_presence[:, neuron_idx_from1]
        return np.nonzero(existing_annotations)[0]

    def present_neurons_at_time(self, t):
        presence = self.neuron_presence[t]
        if presence[0]: presence[0] = False   # should never happen, but just in case, should not send 0 in list of neurons
        return np.argwhere(presence).flatten()

    def get_seg_params(self):
        return self.data.seg_params

    def get_cluster_params(self):
        return self.data.cluster_params

    #this makes the highlight tracks
    def highlighted_track_data(self,highlighted_i_from1):
        # Todo: available for mask data??
        if not self.point_data:
            return [[], []]
        trax=[]
        tray=[]
        for i in range(max(0,self.i+self.tr_pst),min(self.frame_num,self.i+self.tr_fut+1)):
            existing_neurons = np.logical_not(np.isnan(self.NN_or_GT[i][:,0]))
            if existing_neurons[highlighted_i_from1]:
                pt=self.NN_or_GT[i][highlighted_i_from1]
                trax.append(pt[0])
                tray.append(pt[1])
        return np.array([trax,tray])

    def neuron_color(self, idx_from1=None):
        """
        :param idx_from1: if None,
        :return: single list/tuple [r, g, b] the 0-255 RGB values for the color of the neuron if idx_from1 is given,
            otherwise list of such RGB values for each neuron of self.assigned_sorted_list that is currently present
        """
        if idx_from1 is None:
            present = np.argwhere(self.neuron_presence[self.i, self.assigned_sorted_list]).flatten()
            col_lst = [self.color_manager.color_for_neuron(self.assigned_sorted_list[pres_idx]) for pres_idx in present]
            return col_lst
        return self.color_manager.color_for_neuron(idx_from1)
