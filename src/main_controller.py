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
import time
import h5py

#Internal classes
from .helpers import SubProcManager
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

# SJR: message box for indicating neuron number of new neuron and for renumbering neuron
from .msgboxes import EnterCellValue as ecv



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
    """
    # TODO AD: update hlab calcium when changing masks?? (not sure we really use it)
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
        print("Loading dataset:",self.data.name)

        self.ready = False

        # whether data is going to be as points or as masks:
        self.point_data = self.data.point_data

        #Harvard lab
        self.hlab = HarvardLab.HarvardLab(self.data,self.settings)   # TODO: adapt HarvardLab to DataSet

        self.frame_num = self.data.frame_num
        self.data_name = self.data.name

        #we fetch the useful information of the dataset, the h5 file is initialized here.
        self.channel_num = self.data.nb_channels
        self.n_neurons = self.data.nb_neurons
        self.frame_shape = self.data.frame_shape      # Todo: make sure that changes in shae due to cropping do not matter

        self.NNmask_key=""
        self.NNpts_key=""

        # all points are loaded in memory
        if self.point_data:
            self.pointdat = self.data.pointdat
        else:
            self.pointdat = np.full((self.frame_num,self.n_neurons+1,3),np.nan)
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
        self.options["overlay_act"]=True# Todo: should depend on point_data?
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
        self.options["generate_deformation"] = False
        self.options["use_old_trainset"] = False
        self.options["boxing_mode"] = False
        self.options["ShowDim"] = True


        self.tr_fut=5
        self.tr_pst=-5
        self.options["autocenter"]=True
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

        self.button_keys = {}   # TODO AD: rename

        self.assigned_sorted_list = []  # AD sorted list of neurons (from1) that have a key assigned
        self.mask_temp=None

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
        # here when the highlighted neuron changes
        self.highlighted_neuron_registered_clients = []
        # here when the frame content changes
        self.frame_img_registered_clients = []
        # here when the mask changes
        self.mask_registered_clients = []
        # here when any of the pointdat changes
        self.points_registered_clients = []
        # here when the links between points changes
        self.pointlinks_registered_clients = []
        # here when the highlighted track data changes
        self.highlighted_track_registered_clients = []
        # here when the z-viewing should change:
        self.zslice_registered_clients = []   # Todo: Warning, controller will not be aware of z changes if they are done
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

        # list of things that change when the time frame changes:
        # time_frame, frame_img, present_neurons, mask, all kinds of pointdats,

        # EPFL lab specific processes
        self.segmenter = None
        self.ref_frames = set()   # the set of frames that the user wants to use for the next registration
        GlobalParameters.set_params()

    def set_point_data(self, value:bool):
        self.data.point_data = value
        self.point_data = value

    def set_up(self):
        # now we actually initialize
        self.select_frames()
        self.ready = True
        self.update(t_change=True)
        self.signal_nb_neurons_changed()

    def update(self,t_change=False):
        if not self.ready:   # this is jsut to avoid gui elements from calling callbacks resulting in update during their init   # Todo AD: find more elegant way
            return
        # TODO AD: maybe split into smaller methods, and replace calls to update by methods updating only some parts

        # save at update if autosave
        if self.options["autosave"]:
            self.save_status()


        # time change event
        if t_change:
            for client in self.frame_registered_clients:
                client.change_t(self.i)

            # SJR: save number of neurons before loading next frame in order to update neuron bar if needed, see below
            old_n_neurons = self.n_neurons
            # SJR: next step deletes the old mask to prevent "undo" from being based on the previous frame
            if self.options["mask_annotation_mode"]:
                self.mask_temp = None

            # SJR: read mask if time changed and not in point mode
            if not self.point_data:
                k=str(self.i)+"/mask"
                if self.data.check_key(k):
                    #MB: so the mask transformation matches those of the frame which are determined by gui's checkboxes
                    self.mask = self.data.get_mask(self.i, force_original=False)
                    self.data.nb_neurons = int(self.mask.max())
                    self.n_neurons = self.data.nb_neurons
                else:
                    self.mask = None
                    self.data.nb_neurons = 0
                    self.n_neurons = 0

        #load the images from the dataset

        if self.channel_num == 2:
            if self.options["first_channel_only"]:
                self.im_rraw = self.data.get_frame(self.i)
                self.im_graw = None
            elif self.options["second_channel_only"]:#MB: switch R and G channel if g is only supposed to show
                self.im_rraw = self.data.get_frame(self.i, col="green")
                self.im_graw = None
            else:
                self.im_rraw = self.data.get_frame(self.i)
                self.im_graw = self.data.get_frame(self.i, col="green")

        else:
            self.im_rraw = self.data.get_frame(self.i)
            self.im_graw = None


        if self.options["ShowDim"]:
            print("Frame dimensions: "+ str(np.shape(self.im_rraw)))#MB check
            self.options["ShowDim"]=False

        for client in self.frame_img_registered_clients:
            client.change_img_data(self.im_rraw, self.im_graw)

        #load the mask, from ground truth or neural network
        #show mask if either the "overlay mask" OR the "mask annotation mode" checkboxes are checked
        if self.options["overlay_mask"] or self.options["mask_annotation_mode"] or self.options["boxing_mode"]:
            k=str(self.i)+"/mask"
            knn="net/"+self.NNmask_key+"/"+str(self.i)+"/predmask"
            if self.data.check_key(k):
                self.mask = self.data[k][...] # SJR: added [...] to get array back, without [...] got a 'h5py._hl.dataset.Dataset' back
            elif self.data.check_key(knn):
                self.mask = self.data[knn][...] # SJR: added [...] (without testing) to see whether we get array instead of weird data type
            else:
                self.mask=None
            '''
            visualizing masks for epfl lab-MB
            I added this part since I wasn't
            sure if get mask funct. works with lab data or not
            '''
            kcoarse=str(self.i)+"/coarse_mask"
            if not self.point_data:
                if self.data.check_key(k) or self.data.check_key(kcoarse):
                    self.mask = self.data.get_mask(self.i, force_original=False)
        else:
            self.mask = None

        '''
        MB: to check the validation set visually. to see only the NN mask
        '''
        if self.data.only_NN_mask_mode:#MB added
            knn="net/"+self.NNmask_key+"/"+str(self.i)+"/predmask"
            if self.data.check_key(knn):
                self.mask = self.data[knn][...]
            else:
                self.mask = None

        for client in self.mask_registered_clients:
            client.change_mask_data(self.mask)
        self.existing_annotations = np.logical_not(np.isnan(self.NN_or_GT[self.i][:,0]))#MB added:needs to be added befor signal point change to avoid error
        self.signal_pts_changed(t_change=t_change)

        # it is important that this be done AFTER self.signal_pts_changed so that self.NN_or_GT is updated
        self.existing_annotations = np.logical_not(np.isnan(self.NN_or_GT[self.i][:,0]))   # TODO AD what about masks??
        self.signal_nb_neurons_changed()#MB added

        present_neurons = np.nonzero(self.existing_annotations)[0]
        for client in self.present_neurons_registered_clients:
            client.change_present_neurons(present_neurons)

        #peak calculation is only when requested
        self.peak_calced=False

        # SJR: reset neuron bar if changing time point (but only in mask mode because Core et al. don't want that in point mode)
        if t_change and not self.point_data and not old_n_neurons == self.n_neurons:
            self.signal_nb_neurons_changed()

    def valid_points_from_all_points(self, points):
        """
        Keeps only the points in points that have valid (non-nan) coordinates
        :param points: (nb_neurons+1)*3 array, the xyz coords of the point of each neuron (possibly nan if not defined)
        :return: n*3 array (n <= nb_neurons), same as above with all nan lines removed
        """
        valids = np.logical_not(np.isnan(points[:, 0]))
        return points[valids]

    def signal_frame_change(self):
        """
        Signals to all clients that the time frame, and subsequently all values that change along with it such as the
        image and the pointdat, have changed.
        """
        # TODO AD : call all updates
        raise NotImplementedError

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
        st = time.time()
        self.update(t_change=True)
        print("total frame change time:", time.time() - st)

    def go_to_frame(self, t):
        """Moves to frame t and updates everything accordingly"""
        # set the absolute time
        if t < 0 or t >= self.frame_num:
            return
        self.i = t
        self.update(t_change=True)

    def signal_pts_changed(self, t_change=True):
        """
        Updates self.NN_or_GT to match changes in self.pointdat and self.NN_pointdat,
        then re-sends all types of point data.
        Not all may be necessary to re-send in all cases, but this is still better than before (since only the update
        method existed, and would update not only the points but also the image etc)
        """
        # Todo AD: could simplify if only one point was changed...
        if not self.point_data:
            return

        self.NN_or_GT = np.where(np.isnan(self.pointdat), self.NN_pointdat, self.pointdat)

        if self.options["follow_high"] and t_change:
            self.center_on_highlighted()

        if self.options["overlay_tracks"] and self.highlighted != 0:
            for client in self.highlighted_track_registered_clients:
                client.change_track(self.highlighted_track_data(self.highlighted))
                # TODO AD: set =np.zeros((2,0)) = np.zeros((2,0)) otherwise, and do what if not self.point_data?

        pts_dict = {}
        # TODO AD: good init for all pointdats if not self.point_data

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
        #pts_dict["pts_high"] = np.array(self.NN_or_GT[self.i][self.highlighted])[None, :]#MB removed
        pts_dict["pts_high"] = self.valid_points_from_all_points(np.array(self.NN_or_GT[self.i][self.highlighted])[None, :])#MB added
        # TODO AD: set to np.array([[], [], []]).transpose() if not self.point_data, or not define?

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

    def signal_nb_neurons_changed(self):
        """
        Method to signal to all registered clients that the neumber of neurons has changed.
        """
        for client in self.nb_neuron_registered_clients:
            client.change_nb_neurons(self.n_neurons)

    def frame_clicked(self, button, coords):
        # a click with coordinates is received
        if self.options["defining_cropzone_mode"]:
            self.crop_points.append(coords[:2])
        elif self.options["mask_annotation_mode"]:

            """
            SJR: add new neuron / object
            """
            coord = np.array(coords).astype(np.int16)
            # SJR: User pressed the middle button (wheel) = 4, which sets a threshold
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
                    self.data.save_mask(self.i, self.mask, False, True)
                    self.update()

                    self.n_neurons = self.data.nb_neurons
                    self.signal_nb_neurons_changed()
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
                self.data.save_mask(self.i, self.mask, False, True)
                self.update()

                self.n_neurons = self.data.nb_neurons
                self.signal_nb_neurons_changed()
                self.options["RenumberComp"] = False


            # SJR: User left clicks (arg[0]==1) to select which cell to delete
            elif button == 1 and not self.options["boxing_mode"]:
                if not (0 <= coord[0] < self.frame_shape[0] and 0 <= coord[1] < self.frame_shape[1] and 0 <= coord[2] < self.frame_shape[2]):
                    return
                sel = self.mask[coord[0], coord[1], coord[2]]
                if sel != 0:
                    # SJR: simulate clicking on the neuron bar button for selected neuron
                    self.highlight_neuron(sel)
                else:
                    self.highlight_neuron(0)
        #MB: to ba able to draw boxes around objects of interest
        if self.options["boxing_mode"]:
            w,h,d,box_id = self.box_details
            coord = np.array(coords).astype(np.int16)
            if button == 1 and not self.options["RenumberComp"]:#left clicks are only accepted
                if not (coord[0] >= 0 and coord[0] < self.frame_shape[0] and coord[1] >= 0 and coord[1] < self.frame_shape[1] and coord[
                    2] >= 0 and coord[2] < self.frame_shape[2]):
                    return
                self.mask_temp= self.mask.copy()#save to allow undo

                self.mask[coord[0]:coord[0]+w,coord[1]:coord[1]+h,coord[2]:coord[2]+d] = box_id
                self.data.save_mask(self.i, self.mask, False, True)
                self.update()

                self.n_neurons = self.data.nb_neurons
                self.signal_nb_neurons_changed()
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
                self.data.save_mask(self.i, self.mask, False, True)
                self.update()

                self.n_neurons = self.data.nb_neurons
                self.signal_nb_neurons_changed()
                self.options["RenumberComp"] = False

        elif None not in coords:  # skip when the coordinate is NaN
            coord = np.array(coords).astype(np.int16)
            # this will click on the nearest existing annotation
            assert self.point_data, "Not available with mask data."
            indarr = np.nonzero(self.existing_annotations)[0]
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
        coords = np.array(coords)
        if key == "d":
            assert self.point_data, "Not available in absence of point data."
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
        if self.options["mask_annotation_mode"]:
            pass
        else:
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
        # AD
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

    def import_mask_from_external_file(self,Address,green=False):
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
        for t in range(ExtFile.dataset.attrs["T"]):
            mkey = str(t) + "/mask"
            #Assuming the imported file was a derivative of the current file after cropping, rotating and other preprocesses,
            #fr t of imported file corresponds to frame "orig_index" of the current file
            origFrkey = "{}/original_fr".format(t)#original number of frame that t corresponds to is saved in this dataset
            if origFrkey in ExtFile.dataset.keys():
                origIndex = ExtFile.dataset[origFrkey][...]
            else:
                print("The map between the two files frame numbers is not found. It is taken to be identity")
                origIndex = t
            origfrkey = str(origIndex) + "/frame"
            if mkey in ExtFile.dataset.keys() and transformBack and origfrkey in self.data.dataset.keys():
                maskTemp = ExtFile.get_mask(t,force_original=True)#don't apply crop and rotate on the imported masks
                ExtFile.crop = True
                ExtFile.align = True

                img = maskTemp

                key = "{}/transfo_matrix".format(t)
                keyOrig = "{}/transfo_matrix".format(origIndex)
                #copy the cropping parameters of the imported file temporarily into the current file:
                if "ROI" in list(self.data.dataset.attrs.keys()) and keyOrig in list(self.data.dataset.keys()):
                    origROI = self.data.dataset.attrs["ROI"]
                    origTrans = self.data.dataset[keyOrig]
                    TemptransParam = 1
                else:
                    TemptransParam = 0
                self.data.dataset.attrs["ROI"] = ExtFile.dataset.attrs["ROI"]

                if keyOrig not in self.data.dataset:
                    self.data.dataset.create_dataset(keyOrig, data=ExtFile.dataset[key])
                else:
                    self.data.dataset[keyOrig][...] = ExtFile.dataset[key]
                OrigCrop = self.data.crop
                OrigAlign =self.data.align
                self.data.crop = True#so that the save_mask function applies the reverse transformations
                self.data.align = True

                if not np.shape(img)[2] == shapeCheck[2]:
                    print("Z dimensions doesn't match. Zero entries are added to mask for compensation")
                    Zvec = ExtFile.dataset.attrs["original_Z_interval"]
                    imgT = np.zeros((np.shape(img)[0],np.shape(img)[1],shapeCheck[2]))
                    imgT[:,:,int(Zvec[0]):int(Zvec[1])] = img#np.shape(img)[2]] = img
                    if green:
                        self.data.save_green_mask(origIndex, imgT, False, True)
                    else:
                        self.data.save_mask(origIndex, imgT, False, True)
                else:
                    if green:
                        self.data.save_green_mask(origIndex, img, False, True)
                    else:
                        self.data.save_mask(origIndex, img, False, True)
                if TemptransParam == 1:
                    self.data.dataset[keyOrig][...] = origTrans
                    self.data.dataset.attrs["ROI"] = origROI
                self.data.align = OrigAlign
                self.data.crop = OrigCrop
            elif mkey in ExtFile.dataset.keys() and origfrkey in self.data.dataset.keys():
                origFrkey = "{}/original_fr".format(t)#original number of frame that t corresponds to is saved in this dataset
                if origFrkey in ExtFile.dataset.keys():
                    origIndex = ExtFile.dataset[origFrkey][...]
                else:
                    print("The map between the two files frame numbers is not found. It is taken to be identity")
                    origIndex = t
                img = ExtFile.get_mask(t,force_original=True)
                if not np.shape(img)[2] == shapeCheck[2]:
                    print("Z dimensions doesn't match. Zero entries are added to mask for compensation")
                    Zvec = ExtFile.dataset.attrs["original_Z_interval"]
                    imgT = np.zeros((np.shape(img)[0],np.shape(img)[1],shapeCheck[2]))
                    imgT[:,:,int(Zvec[0]):int(Zvec[1])] = img
                    if green:
                        self.data.save_green_mask(origIndex, imgT, True, True)
                    else:
                        self.data.save_mask(origIndex, imgT, True, True)

                else:
                    if green:
                        self.data.save_green_mask(origIndex, img, True, True)
                    else:
                        self.data.save_mask(origIndex, img, True, True)
        ExtFile.close()
        print("mask upload finished")

    def Preprocess_and_save(self,frame_int,frame_deleted,Z_interval,X_interval,Y_interval,bg_blur,sd_blur,bg_subt,width,height):
        """
        MB defined this to select and delete the desired frames from the
        original movie or blur, subtract background andcrop in z direction.
        it saves the result in a new .h5 file in the directory of the original input files
        """
        self.save_status()
        self.update()


        frameCheck = self.data.get_frame(0, col= "red")
        Zcheck = np.shape(frameCheck)


        z_0 = int(Z_interval[0])
        z_1 = int(Z_interval[1])

        padXL = 0
        padXR = 0
        padYbottom = 0
        padYtop = 0

        if int(X_interval[0])<0:
            x_0 = 0
            padXL = 0-int(X_interval[0])
        else:
            x_0 = int(X_interval[0])

        if int(X_interval[1])==0:
            x_1 = Zcheck[0]
        elif int(X_interval[1])>Zcheck[0]:
            x_1 = Zcheck[0]
            padXR = int(X_interval[1])-Zcheck[0]
        else:
            x_1 = int(X_interval[1])

        if int(Y_interval[0]) <0:
            y_0 = 0
            padYbottom = 0-int(Y_interval[0])
        else:
            y_0 = int(Y_interval[0])

        if int(Y_interval[1])==0:#doesn't change x and y coordinate if the upper bound is zero
            y_1 = Zcheck[1]
        elif int(Y_interval[1])>Zcheck[1]:
            y_1 = Zcheck[1]
            padYtop = int(Y_interval[1])-Zcheck[1]
        else:
            y_1 = int(Y_interval[1])

        assert z_0 < Zcheck[2], "lower bound for z is too high"
        assert not z_1 > Zcheck[2], "upper bound for z is too high"
        assert x_0 < Zcheck[0], "lower bound for x is too high"
        assert not x_1 > Zcheck[0], "upper bound for x is too high"





        dset_path=self.data.dset_path_from_GUI
        name=self.data.name#TODO: get it as input
        dset_path_rev = dset_path[::-1]
        key=name+"_CroppedandRotated"
        if '/' in dset_path_rev:
            SlashInd = dset_path_rev.index('/')
            dset_path_cropped = dset_path[0:len(dset_path)-SlashInd]
            newpath = os.path.join(dset_path_cropped,key+".h5")
        else:
            newpath = key+".h5"
        fd = h5py.File(newpath, 'w')
        for a in self.data.dataset.attrs:
            fd.attrs[a] = self.data.dataset.attrs[a]
        hNew = DataSet.load_dataset(newpath)
        OrigCrop = self.data.crop
        OrigAlign =self.data.align
        if self.options["save_crop_rotate"]:
            self.data.crop = True
            self.data.align = True
        l=0

        key="distmat"
        if key in self.data.dataset.keys():
            distmat = self.data.dataset[key]
            ds = hNew.dataset.create_dataset(key,shape=np.shape(distmat),dtype="f4")
            ds[...]=distmat
            print("distmat saved")

        for i in frame_int:
            if i not in frame_deleted and i in self.selected_frames:
                if self.options["AutoDelete"]:
                    kcoarse=str(i)+"/coarse_mask"
                    if kcoarse in self.data.dataset.keys():
                        if len(np.unique(self.data.dataset[kcoarse]))<3:
                            continue


                if hNew.dataset.attrs["C"]==2:
                    if (self.options["save_1st_channel"] and self.options["save_green_channel"]) :
                        frameGr = self.data.get_frame(i, col= "green")
                        frameGr = frameGr[:x_1,:y_1,:z_1]
                        frameGr = frameGr[x_0:,y_0:,z_0:]

                        frameGr = np.pad(frameGr, ((padXL, padXR),(padYtop, padYbottom),(0,0)),'constant', constant_values=((0, 0),(0,0), (0,0)))

                        if self.options["save_resized"]:
                            frameGr = self.resize_frame(frameGr,width,height)
                    elif self.options["save_1st_channel"] or self.options["save_green_channel"]:
                        frameGr = 0
                    else:
                        frameGr = self.data.get_frame(i, col= "green")
                        frameGr = frameGr[:x_1,:y_1,:z_1]
                        frameGr = frameGr[x_0:,y_0:,z_0:]
                        frameGr = np.pad(frameGr, ((padXL, padXR),(padYtop, padYbottom), (0,0)),'constant', constant_values=((0, 0),(0,0), (0,0)))
                        if self.options["save_resized"]:
                            frameGr = self.resize_frame(frameGr,width,height)
                else:
                    frameGr = 0

                if self.options["save_green_channel"] and not self.options["save_1st_channel"]:
                    frameRd = self.data.get_frame(i, col= "green")
                else:
                    frameRd = self.data.get_frame(i, col="red")
                if self.options["save_blurred"]:
                    frameRd = self.Blur(frameRd, bg_blur, sd_blur, self.options["save_subtracted_bg"],bg_subt)
                elif self.options["save_subtracted_bg"]:
                    frameRd = self.SubtBg(frameRd,bg_subt)
                frameRd = frameRd[:x_1,:y_1,:z_1]
                frameRd = frameRd[x_0:,y_0:,z_0:]
                frameRd = np.pad(frameRd, ((padXL, padXR),(padYtop, padYbottom),  (0,0)),'constant', constant_values=((0, 0),(0,0), (0,0)))
                if self.options["save_resized"]:
                    frameRd = self.resize_frame(frameRd,width,height)

                mkey = str(i) + "/mask"
                if mkey in self.data.dataset.keys():
                    maskTemp = self.data.get_mask(i)
                    maskTemp = maskTemp[:x_1,:y_1,:z_1]
                    maskTemp = maskTemp[x_0:,y_0:,z_0:]
                    maskTemp = np.pad(maskTemp, ( (padXL, padXR),(padYtop, padYbottom), (0,0)),'constant', constant_values=((0, 0),(0,0), (0,0)))
                    fd.attrs["N_neurons"] = np.maximum(fd.attrs["N_neurons"],np.maximum(len(np.unique(maskTemp)),np.max(maskTemp)))
                    if self.options["save_resized"]:
                        maskTemp = self.resize_frame(maskTemp,width,height,mask=True)
                    hNew.save_frame(l, frameRd, frameGr, mask = maskTemp , force_original=True)
                else:
                    hNew.save_frame(l,frameRd, frameGr ,force_original=True)
                kcoarse=str(i)+"/coarse_mask"
                if False:#kcoarse in self.data.dataset.keys() and not self.options["save_crop_rotate"]:#TODO : check for compatibility with rotation and cropping modes
                    CoarseSegTemp = self.data.dataset[kcoarse]
                    CoarseSegTemp = CoarseSegTemp[:x_1,:y_1,:z_1]
                    CoarseSegTemp = CoarseSegTemp[x_0:,y_0:,z_0:]
                    CoarseSegTemp = np.pad(CoarseSegTemp, ((padXL, padXR),(padYtop, padYbottom), (0,0)),'constant', constant_values=((0, 0),(0,0),(0,0)))
                    kcoarsel=str(l)+"/coarse_mask"
                    kcoarseSegl=str(l)+"/coarse_seg"
                    hNew.dataset.create_dataset(kcoarsel, data=CoarseSegTemp)
                    hNew.dataset.create_dataset(kcoarseSegl, data=CoarseSegTemp)
                    print(i)
                #save the transformation functions for later retrieval
                key = "{}/transfo_matrix".format(i)
                keyl = "{}/transfo_matrix".format(l)
                if self.options["save_crop_rotate"] or (key in self.data.dataset.keys()):
                    matrix = self.data.dataset[key]
                    print(i)
                    if key not in hNew.dataset:
                        hNew.dataset.create_dataset(keyl, data=matrix)
                    else:
                        hNew.dataset[keyl][...] = matrix
                keyfrnum = "{}/original_fr".format(l)
                hNew.dataset.create_dataset(keyfrnum, data=int(i))
                keyTime = "{}/time".format(l)
                RealTimeKey = "{}/time".format(i)
                if RealTimeKey in self.data.dataset.keys():
                    realTime = np.array(self.data.dataset[RealTimeKey])
                    hNew.dataset.create_dataset(keyTime, data=realTime)
                l=l+1
        frameRd_shape = np.shape(frameRd)
        fd.attrs["W"] = frameRd_shape[0]
        fd.attrs["H"] = frameRd_shape[1]
        if self.options["save_resized"]:
            hNew.dataset.attrs["Original_size"] = [Zcheck[0],Zcheck[1]]
            fd.attrs["W"] = int(width)
            fd.attrs["H"] = int(height)

        if self.options["save_crop_rotate"] or ("ROI" in self.data.dataset.attrs.keys()):
            hNew.dataset.attrs["ROI"] = self.data.dataset.attrs["ROI"]
        hNew.dataset.attrs["original_Z_interval"] = Z_interval
        hNew.dataset.attrs["original_X_interval"] = X_interval
        hNew.dataset.attrs["original_Y_interval"] = Y_interval

        if self.options["save_1st_channel"] and not self.options["save_green_channel"]:
            fd.attrs["C"] = 1
        elif not self.options["save_1st_channel"] and self.options["save_green_channel"]:
            fd.attrs["C"] = 1
        fd.attrs["T"] = l
        fd.attrs["D"] = z_1-z_0
        self.data.align = OrigAlign
        self.data.crop = OrigCrop




        hNew.close()


    def Blur(self,frame,blur_b=40,blur_s=6, Subt_bg=False,subtVal = 1):
        """this blures the images and subtracts background if asked"""
        dimensions=(.1625, .1625, 1.5)
        sigm=blur_s#value between 1-10
        bg_factor=blur_b#value between 0-100
        xysize, xysize2, zsize = dimensions
        sdev = np.array([sigm, sigm, sigm * xysize / zsize])
        im_rraw = frame
        im = im_rraw
        sm = sim.gaussian_filter(im, sigma=sdev) - sim.gaussian_filter(im, sigma=sdev * bg_factor)
        im_rraw = sm
        if Subt_bg:
            threshold_r=(im_rraw<subtVal)
            im_rraw[threshold_r]=0

        frame=im_rraw

        return frame

    def SubtBg(self,frame,subtVal):
        """subtracts a constant background valuefrom your movie"""
        im_rraw = frame
        threshold_r=(im_rraw<subtVal)
        im_rraw[threshold_r]=0
        return im_rraw

    def resize_frame(self, frame, width,height, mask=False):
        """resizes the frame to the dimensions given as width and height"""
        #width = 16*int(width/16)
        #height = 16*int(height/16)
        frameResize = np.zeros((width,height,np.shape(frame)[2]))
        for j in range(np.shape(frame)[2]):
            if mask:
                frameResize[:,:,j] = cv2.resize(frame[:,:,j], dsize=(height, width), interpolation=cv2.INTER_NEAREST)
            else:
                frameResize[:,:,j] = cv2.resize(frame[:,:,j], dsize=(height, width), interpolation=cv2.INTER_CUBIC)
        return frameResize

    def highlight_neuron(self, neuron_id_from1):
        """
        Changes the highlighted neuron. If neuron_id_from1 was already highlighted, unhiglight it.
        Otherwise, unhighlight the highlighted neuron (if any) and highlight neuron_id_from1.
        """

        if neuron_id_from1 == self.highlighted:
            self.highlighted = 0
            for client in self.highlighted_neuron_registered_clients:
                client.change_highlighted_neuron(unhigh=neuron_id_from1)
        else:
            hprev = self.highlighted
            self.highlighted = neuron_id_from1
            for client in self.highlighted_neuron_registered_clients:
                client.change_highlighted_neuron(high=self.highlighted,
                                                 unhigh=(hprev if hprev else None),
                                                 high_present=bool(not self.point_data or self.existing_annotations[self.highlighted]),
                                                 unhigh_present=bool(not self.point_data or self.existing_annotations[hprev] if hprev else False),
                                                 high_key=self._get_neuron_key(self.highlighted))
        self.update()#MB added for solving points highlightimg issue. maybe instead, one could add Mainfigwidget to .highlighted_neuron_registered_clients

    # this assigns the neuron keys without overlap
    def assign_neuron_key(self, i_from1, key):
        key_changes = []   # list of (ifrom1, newkey) that need to be changed
        if key in self.button_keys:
            i_from1_prev = self.button_keys.pop(key)
            key_changes.append((i_from1_prev, None))

        if i_from1 in self.button_keys.values():
            key_prev = list(self.button_keys.keys())[list(self.button_keys.values()).index(i_from1)]
            self.button_keys.pop(key_prev)

        if len(self.button_keys) == int(self.settings["max_sim_tracks"]):
            errdial = QErrorMessage()   # TODO AD
            errdial.showMessage("Too many keys set(>" + str(len(self.button_keys)) + "). Update settings if needed")
            errdial.exec_()
            return

        self.button_keys[key] = i_from1
        key_changes.append((i_from1, key))
        for client in self.neuron_keys_registered_clients:
            client.change_neuron_keys(key_changes)
        print("Assigning key:", key, "for neuron", i_from1)

        self.assigned_sorted_list = sorted(list(self.button_keys.values()))

        self.set_activated_tracks()   # TODO

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
        if self.existing_annotations[self.highlighted]:
        # set z to the highlighted neuron
            for client in self.zslice_registered_clients:
                client.change_z(int(self.NN_or_GT[self.i][self.highlighted, 2] + 0.5))

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
        self.signal_pts_changed(t_change=False)
        self.update()

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
        self.update(t_change=False)   # TODO: not update everything

    def toggle_NN_mask_only(self): #MB added to check different NN results
        self.data.only_NN_mask_mode = not self.data.only_NN_mask_mode
        self.update(t_change=True)


    def toggle_display_alignment(self):
        self.data.align = not self.data.align
        self.update(t_change=True)  # todo: could be just update()?

    def toggle_display_cropped(self):
        self.data.crop = not self.data.crop
        self.update(t_change=True)  # todo: could be just update()?

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
        # update number of neurons and neuron bar
        self.n_neurons = self.data.nb_neurons
        self.signal_nb_neurons_changed()
        # Todo: we might also want to update masks?

    def extract_features(self):
        feature_builder = FeatureBuilder(self.data)
        feature_builder.extract_features(self.selected_frames)

    def cluster(self):
        clustering = Clustering(self.data, self.data.cluster_params)
        clustering.find_assignment(self.selected_frames)  # MB changed from data.frames to selected_frames
        # update number of neurons and neuron bar
        self.n_neurons = self.data.nb_neurons
        self.signal_nb_neurons_changed()

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
        print("The non-gound-truth frames you are classifying:")  # MB added
        print(set(self.data.segmented_non_ground_truth()))
        clf.find_assignment(self.data.segmented_non_ground_truth())

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
        print("Left click on the lower left corner of the box you like to insert")
        self.update()

    def toggle_mask_annotation_mode(self):
        assert not self.point_data, "Not available for point data."
        self.options["mask_annotation_mode"] = not self.options["mask_annotation_mode"]
        if self.options["mask_annotation_mode"]:
            print("Mask Annotation Mode")
        else:
            self.mask_temp = None
            print("Saving Last Push")
        self.update()   # TODO AD: is this necessary?

    def set_mask_annotation_threshold(self, value):   # SJR
        """Set mask threshold"""
        self.mask_thres = int(value)
        for client in self.mask_thres_registered_clients:
            client.change_mask_thresh(self.mask_thres)

    def set_box_dimensions(self, info): #MB
        """ To help annotation with boxes. updates the dimensions of the box and the cell id we want to assign to it"""
        print(info)
        box_info=info.split('-')
        dimensions = box_info[0].split(',')
        box_id = box_info[1]
        print(dimensions)
        assert len(dimensions)==3, "Dimensions of the box is not correct"
        self.box_details = [int(dimensions[0]), int(dimensions[1]), int(dimensions[2]), int(box_id)]

    def renumber_mask_obj(self):
        if self.highlighted == 0:
            return
        #MB added: to get the connected components of the mask
        labelArray,numFtr = sim.label(self.mask==self.highlighted)
        if numFtr>1:
            dlg2 = ecv.CustomDialogSubCell()#ask if you want all components change or ony one
        #ind=regs[0][coord[0],coord[1],coord[2]]

        dlg = ecv.CustomDialog()#which number to change to

        # if the user presses 'ok' in the dialog window it executes the code
        # else it does nothing
        # it also tests that the user has entered some value, that it is not
        # empty and that it is equal or bigger to 0.
        if dlg.exec_() and dlg.entry1.text() != '' and int(dlg.entry1.text()) >= 0:

            # reads the new value to set and converts it from str to int
            value = int(dlg.entry1.text())

            # SJR: remember how many neurons there were
            temp_n_neurons = self.n_neurons
            # SJR: save old mask to allow undo
            self.mask_temp = self.mask.copy()
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
            self.data.save_mask(self.i, self.mask, False, True)
            self.n_neurons = self.data.nb_neurons
            if temp_n_neurons != self.n_neurons:
                self.signal_nb_neurons_changed()
            # unhighlight and turn off neuron_bar button, careful!! does an update, which resets self.mask
            self.highlight_neuron(self.highlighted)

        self.update()

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
                temp_n_neurons = self.n_neurons
                for k in range(fro,to):
                    key=str(k)+"/mask"
                    if self.data.check_key(key):
                        mask_k = self.data.get_mask(k, force_original=False)#MB added
                        mask_k[mask_k == self.highlighted] = value
                        self.data.save_mask(k, mask_k, False, True)

                self.n_neurons = self.data.nb_neurons
                if temp_n_neurons != self.n_neurons:
                    self.signal_nb_neurons_changed()
                self.highlight_neuron(self.highlighted)
        self.update()

    def permute_masks(self, Permutation):
        temp_n_neurons = self.n_neurons
        self.mask_temp = self.mask.copy()
        for l in range(len(Permutation)-1):
            k = Permutation[l]
            print(k)
            self.mask[self.mask_temp == k] = Permutation[l+1]
            self.data.save_mask(self.i, self.mask, False, True)
            self.n_neurons = self.data.nb_neurons
            if temp_n_neurons != self.n_neurons:
                self.signal_nb_neurons_changed()
            #self.highlight_neuron(self.highlighted)
        self.update()

    def delete_mask_obj(self):
        if self.highlighted == 0:
            return
        # SJR: remember how many neurons there were
        temp_n_neurons = self.n_neurons
        # SJR: save old mask to allow undo
        self.mask_temp = self.mask.copy()
        # SJR: erase neuron
        self.mask[self.mask == self.highlighted] = 0
        self.data.save_mask(self.i, self.mask, False, True)
        print("SJR: self.mask.max() in delete_mask_obj", self.mask.max())
        # unhighlight and turn off neuron_bar button, careful!! does an update, which resets self.mask
        self.n_neurons = self.data.nb_neurons
        if temp_n_neurons != self.n_neurons:
            self.signal_nb_neurons_changed()
        self.highlight_neuron(self.highlighted)
        print("SJR: self.mask.max() in delete_mask_obj", self.mask.max())
        print("SJR: self.data.nb_neurons in delete_mask_obj", self.data.nb_neurons)

    def undo_mask(self):
        if self.options["mask_annotation_mode"] or self.options["boxing_mode"]:
            if self.mask_temp is not None:
                self.mask = self.mask_temp
                self.data.save_mask(self.i, self.mask)
                self.update()

    ####################################################################################################################
    # self.hlab and ci related methods

    def update_ci(self):
        self.hlab.update_all_ci(self.data)

    ####################################################################################################################
    # NN related methods

    def clear_frame_NN(self):
        """Deletes all NN predictions in current frame"""
        # DEPRECATED, TODO CFP?
        print("Deleting all annotations in frame", self.i)
        self.pointdat[self.i] = np.nan
        self.NN_pointdat[self.i] = np.nan
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
        if not self.point_data:#MB added this to use this feature for epfl data
            temp_n_neurons = self.n_neurons
            for k in range(fro,to):
                key=str(k)+"/mask"
                if self.data.check_key(key):
                    mask_k = self.data.get_mask(k, force_original=False)#MB added
                    mask_k[mask_k == self.highlighted] = 0
                    self.data.save_mask(k, mask_k, False, True)
            self.n_neurons = self.data.nb_neurons
            if temp_n_neurons != self.n_neurons:
                self.signal_nb_neurons_changed()
            self.highlight_neuron(self.highlighted)

        self.update()

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
        self.update()

    def select_NN_instance_points(self, NetName:str, instance:str):
        """Loads neural network point predictions"""
        self.save_status()
        if not self.point_data:
            print("No points")
            return
        if NetName == "":
            self.NNpts_key = ""
            self.NN_pointdat = np.full_like(self.pointdat, np.nan)
            self.update()
            return
        key = NetName + "_" + instance
        knn = "net/" + key + "/NN_pointdat"
        if self.data.check_key(knn):
            self.NNpts_key = knn
            self.NN_pointdat = np.array(self.data[self.NNpts_key])
            self.NN_pointdat[:, 0, :] = np.nan
        else:
            self.NN_pointdat = np.full_like(self.pointdat, np.nan)
        self.update()

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
                mkey = str(t) + "/mask"
                if True:#not mkey in self.data.dataset.keys():
                    knn="net/"+self.NNmask_key+"/"+str(t)+"/predmask"
                    if knn in self.data.dataset.keys():
                        mask = self.data[knn][...]
                        self.data.save_mask(t, mask, False, True)
                    else:
                        print("There are no predictions for this frame")
        self.update()

    def post_process_NN_masks(self,ExemptNeurons):
        """MB added: to post process the predicttions of NN for the selected frames as the ground truth
        ExemptNeurons: neurons that you do not want to postprocess

        """
        if self.NNmask_key == "":
            print("You should first choose the NN instance")
        else:
            for t in self.selected_frames:
                mkey = str(t) + "/mask"
                if True:#not mkey in self.data.dataset.keys():
                    knn="net/"+self.NNmask_key+"/"+str(t)+"/predmask"
                    if knn in self.data.dataset.keys():
                        mask = self.data[knn][...]
                        if True:#for c in cell_list:
                            labelArray, numFtr = sim.label(mask>0)#get all the disconnected components of the nonzero regions of mask
                            for i in range(numFtr+1):
                                IfZero = False
                                submask =  (labelArray==i)#focus on each connected component separately
                                list = np.unique(mask[submask])#list of all cell ids corresponded to the connected component i
                                list = [int(k) for k in (set(list)-set(ExemptNeurons))]
                                if np.sum(submask)<3:#if the component is only 1 or 2 pixels big:
                                    for l in range(len(list)):
                                        if np.sum(mask==list[l])>5:#check if this is the only place any cell is mentioned. if a cell is mentioned somewhere else set this component to zero.
                                            if list[l] not in ExemptNeurons:
                                                mask[submask]=0
                                                IfZero = True #whether the component was set to zero

                                elif len(list)>1 and not IfZero:
                                    Volume = np.zeros(len(list))
                                    for l in range(len(list)):
                                        Volume[l] = sum(mask[submask]==list[l])
                                    BiggestCell = list[np.argmax(Volume)]
                                    mask[submask] = BiggestCell
                        self.data[knn][...] = mask
                    else:
                        print("There are no predictions for this frame")
        self.update()

    def post_process_NN_masks2(self,ExemptNeurons):
        """MB added: to post process the predicttions of NN for the selected frames as the ground truth
        ExemptNeurons: neurons that you do not want to postprocess.
        This mode uses different connectivity criteria

        """

        if self.NNmask_key == "":
            print("You should first choose the NN instance")
        else:
            s = [[[False, False, False],
                [False,  True, False],
                [False, False, False]],
                [[False,  True, False],
                [ False,  True,  False],
                [False,  True, False]],
                [[False, False, False],
                [False,  True, False],
                [False, False, False]]]
            for t in self.selected_frames:
                mkey = str(t) + "/mask"
                knn="net/"+self.NNmask_key+"/"+str(t)+"/predmask"
                if knn in self.data.dataset.keys():
                    mask = self.data[knn][...]
                    maskOrig = mask
                    labelArray, numFtr = sim.label(mask>0,structure=s)#get all the disconnected components of the nonzero regions of mask
                    Grandlist = np.unique(mask)#list of all cell ids in the mask
                    Grandlist = [int(k) for k in (set(Grandlist)-set(ExemptNeurons))]
                    for c in Grandlist:
                        print(c)
                        #get connected component of each neuron
                        labelArray_c, numFtr_c = sim.label(maskOrig==c,structure=s)#get all the disconnected components of a certain cell class
                        Vol = np.zeros([1,numFtr_c])
                        for i in range(0,numFtr_c):
                            Vol[0,i] = np.sum(labelArray_c==i+1)#volume of each connected component of cell class c
                        print("Vol")
                        print(Vol)
                        BigComp=np.argmax(Vol)+1#label of the biggest component of class
                        print("BigComp")
                        print(BigComp)
                        Comp_c_BigComp = (labelArray_c==BigComp)#biggest connected component labeled as c
                        label_c_BigComp = np.unique(labelArray[Comp_c_BigComp])
                        for i in range(1,numFtr_c+1):
                            print(i)
                            if not i==BigComp:
                                Comp_c_i = (labelArray_c==i)
                                label_c_i = np.unique(labelArray[Comp_c_i])#label of c_i component in the first big labeling
                                print(label_c_i)
                                connCom_containing_c_i = (labelArray==label_c_i)
                                print(np.sum(connCom_containing_c_i))
                                cells_Connected_To_i =set(np.unique(maskOrig[connCom_containing_c_i]))-{c}-{0}
                                cells_Connected_To_i = [int(k) for k in  cells_Connected_To_i]
                                print("cells_Connected_To_i")
                                print(cells_Connected_To_i)
                                if not label_c_i== label_c_BigComp:
                                    if len(cells_Connected_To_i) == 1:#if only one other cell is connected to c_i
                                        mask[Comp_c_i] = cells_Connected_To_i[0]
                                    elif len(cells_Connected_To_i) > 1:
                                        Vol_c_i_conn = np.zeros([1,len(cells_Connected_To_i)])
                                        for j in range(len(cells_Connected_To_i)):
                                            Vol_c_i_conn[0,j] = np.sum(connCom_containing_c_i&(mask==cells_Connected_To_i[j]))
                                            print("Vol_c_i_conn")
                                            print(Vol_c_i_conn)
                                        mask[Comp_c_i] = int(cells_Connected_To_i[np.argmax(Vol_c_i_conn)])
                                    #elif:
                                    #    mask[Comp_c_i] = 0
                    self.data[knn][...] = mask


    def post_process_NN_masks3(self,ProcessedNeurons):
        """MB added: to post process the predicttions of NN for the selected frames as the ground truth
        ProcessedNeurons: neurons that you want to postprocess. If two or three neurons touch each other
         and are in one connected component it renames the smaller ones to the largest one

        """
        Vol = np.zeros([1,len(ProcessedNeurons)])
        if self.NNmask_key == "":
            print("You should first choose the NN instance")
        else:
            for t in self.selected_frames:
                mkey = str(t) + "/mask"
                if True:#not mkey in self.data.dataset.keys():
                    knn="net/"+self.NNmask_key+"/"+str(t)+"/predmask"
                    if knn in self.data.dataset.keys():
                        mask = self.data[knn][...]
                        if True:#for c in cell_list:
                            labelArray, numFtr = sim.label(mask>0)#get all the disconnected components of the nonzero regions of mask
                            for i in range(1,numFtr+1):
                                submask =  (labelArray==i)#focus on each connected component separately
                                for k in range(len(ProcessedNeurons)):
                                    Vol[0,k] = np.sum(mask[submask]==ProcessedNeurons[k])#volume of each of the chosen neurons in this component
                                MaxInd = np.argmax(Vol[0,:])
                                if not np.max(Vol[0,:])==0:
                                    for k1 in range(len(ProcessedNeurons)):
                                        if not k1==MaxInd:
                                            k1_comp = (submask&(mask==ProcessedNeurons[k1]))
                                            mask[k1_comp] = int(ProcessedNeurons[MaxInd])
                        self.data[knn][...] = mask
                    else:
                        print("There are no predictions for this frame")
        self.update()


    def post_process_NN_masks4(self,ProcessedNeurons):
        """MB added: to post process the predicttions of NN for the selected frames as the ground truth
        ProcessedNeurons: neurons that you want to postprocess. if a certain neuron has multiple components
         it deletes the components that have smaller volume

        """

        if self.NNmask_key == "":
            print("You should first choose the NN instance")
        else:
            for t in self.selected_frames:
                mkey = str(t) + "/mask"
                if True:#not mkey in self.data.dataset.keys():
                    knn="net/"+self.NNmask_key+"/"+str(t)+"/predmask"
                    if knn in self.data.dataset.keys():
                        mask = self.data[knn][...]
                        for n in ProcessedNeurons:#for c in cell_list:
                            labelArray, numFtr = sim.label(mask==n)#get all the disconnected components of the nonzero regions of mask
                            if numFtr>1:
                                Vol = np.zeros([1,numFtr])
                                for i in range(1,numFtr+1):
                                    Vol[0,i-1] =  np.sum(labelArray==(i))#focus on each connected component separately
                                print("Vol"+str(Vol))
                                MaxInd = np.argmax(Vol[0,:])
                                print("MaxInd"+str(MaxInd))
                                print("features to delete")
                                for k1 in range(numFtr):
                                    if not k1==MaxInd:
                                        print(k1+1)
                                        k1_comp = (labelArray==(k1+1))
                                        mask[k1_comp] = 0
                        self.data[knn][...] = mask
                    else:
                        print("There are no predictions for this frame")
        self.update()
    def post_process_NN_masks5(self,ProcessedNeurons):
        """MB added: to post process the predicttions of NN for the selected frames as the ground truth
        ProcessedNeurons: neurons that you want to postprocess. If two or three neurons touch each other
         and are in one connected component it renames all to the first neuron in ProcessedNeurons vector

        """
        Vol = np.zeros([1,len(ProcessedNeurons)])
        if self.NNmask_key == "":
            print("You should first choose the NN instance")
        else:
            for t in self.selected_frames:
                mkey = str(t) + "/mask"
                if True:#not mkey in self.data.dataset.keys():
                    knn="net/"+self.NNmask_key+"/"+str(t)+"/predmask"
                    if knn in self.data.dataset.keys():
                        mask = self.data[knn][...]
                        if True:#for c in cell_list:
                            labelArray, numFtr = sim.label(mask>0)#get all the disconnected components of the nonzero regions of mask
                            for i in range(1,numFtr+1):
                                submask =  (labelArray==i)#focus on each connected component separately
                                for k in range(len(ProcessedNeurons)):
                                    Vol[0,k] = np.sum(mask[submask]==ProcessedNeurons[k])#volume of each of the chosen neurons in this component
                                #MaxInd = np.argmax(Vol[0,:])
                                if not np.max(Vol[0,:])==0 and not Vol[0,0]==0:
                                    for k1 in range(len(ProcessedNeurons)):
                                        if not k1==0:
                                            k1_comp = (submask&(mask==ProcessedNeurons[k1]))
                                            mask[k1_comp] = int(ProcessedNeurons[0])
                        self.data[knn][...] = mask
                    else:
                        print("There are no predictions for this frame")
        self.update()


    def check_NN_run(self):
        # checks the subprocess running status
        rundata = self.subprocmanager.check()
        # rundata={"NNtest":[["maskgen",0.1],["train",0.45],["pred",0.58]]}
        dialog = QtHelpers.Dialog(rundata, self)   # TODO AD
        dialog.exec_()
        dialog.deleteLater()

    def run_NN_masks(self, modelname, instancename, fol,epoch,train,validation,targetframes):
        # Todo AD could this be factorized in some way?
        # run a mask prediction neural network
        self.save_status()
        if modelname == "RGN":
            return False, "RGN cannot be used for masks"

        dset_path = self.data.dset_path_from_GUI
        name = self.data.name

        # temporary close
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
        if not self.options["use_old_trainset"] and not self.options["generate_deformation"]:
            args = ["python3", "./src/neural_network_scripts/run_NNmasks_f.py", newpath, newlogpath,"0",str(epoch),"0","0",str(train),str(validation)]
        elif not self.options["use_old_trainset"] and self.options["generate_deformation"]:
            args = ["python3", "./src/neural_network_scripts/run_NNmasks_f.py", newpath, newlogpath,"1","1","1","0","2",str(targetframes)]
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

    def run_NN_points(self, modelname, instancename, fol):
        # Todo AD could this be factorized in some way?
        # run a point prediction neural network
        self.save_status()
        RGN = (modelname == "RGN")
        dset_path = self.data.path_from_GUI
        name = self.data.name
        # temporary close
        key = name + "_" + modelname + "_" + instancename
        newpath = os.path.join("data", "data_temp", key + ".h5")
        newlogpath = os.path.join("data", "data_temp", key + ".log")
        if key in self.subprocmanager.runnings.keys():
            return False, "This run is already running."
        if os.path.exists(newpath):
            return False, "There is an unpulled instance of this run."

        # we are safe now.
        self.data.close()  # close
        shutil.copyfile(dset_path, newpath)
        self.data = DataSet.load_dataset(dset_path)  # reopen
        args = ["python3", "./src/neural_network_scripts/run_NNpts.py", newpath, newlogpath]
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
            shutil.copyfile("./src/neural_network_scripts/run_" + ("RGN" if RGN else "NNpts") + ".py",
                            os.path.join(key, "run_" + ("RGN" if RGN else "NNpts") + ".py"))
            for filename in ["NNtools.py", "FourierAugment.py", "autoencoder.ipynb", "conv_autoenc.py", "UNet2d.py",
                             "highlighter.ipynb"]:
                shutil.copyfile(os.path.join("src", "neural_network_scripts", filename), os.path.join(key, filename))
            with open(os.path.join(key, "run.sh"), "w") as f:
                if RGN:
                    f.write("python3 run_RGN.py" + " " + nnewpath + " " + nnewlogpath)
                else:
                    f.write("python3 run_NNpts.py" + " " + nnewpath + " " + nnewlogpath)
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

    def _scan_NN_instances(self):
        """
        Looks for existing NN instances in self.data, and populates the dict self.NNinstances with them.
        Only used for initialization.
        """
        for key in self.data.available_NNpointdats():   # Todo AD why for pointdat only? is it just the name?
            NetName, instance = key.split("_")
            if NetName not in self.NNinstances:
                self.NNinstances[NetName] = []
            if instance not in self.NNinstances[NetName]:
                self.NNinstances[NetName].append(instance)

    def pull_NN_res(self, key: str, success: bool):
        """
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

            if self.point_data:
                def pointdat_filter_fun(NNmodel, instance):
                    return "NN_pointdat" in self.data["net"][NNmodel + "_" + instance]   # TODO AD self.data["net"]
            else:
                def pointdat_filter_fun(NNmodel, instance):
                    return False

            for client in self.NN_instances_registered_clients:
                client.change_NN_instances(pointdat_filter_fun)
        else:
            print("Deleting ", key)
            val, msg = True, "Deleted"
        os.remove(newlogpath)
        os.remove(newpath)
        self.subprocmanager.free(key)
        return val, msg

    def NN_inst_has_pointdat(self, NNmodel: str, instance: str):
        """
        Filtering function that returns whether the dataset has associated pointdat.
        :param NNmodel: standard NN model name
        :param instance: name of the instance of NNmodel
        :return: bool, whether given instance has associated pointdat in self.data.
        """
        if not self.point_data:
            return False
        return "NN_pointdat" in self.data["net"][NNmodel + "_" + instance]  # TODO AD self.data["net"]

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

            ws=self.imdat.swapaxes(0,1)[pts[0],pts[1],pts[2]]   # Todo: is self.imdat defined??

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
        if self.point_data:
            self.save_pointdat()
            self.save_NNpointdat()
            self.hlab.save_ci_int(self.data)#MB just added a tab to avoid an error with mask data
        self.data.save()
        # Todo AD: if th ci_int changes, we might want to update the ci display (which used to be done by calling
        #  self.update() after self.save_status() everytime, but that was a bit overkill...)
        print("Saved")

    #we save the point data
    def save_pointdat(self):
        # TODO: check for point_data use and consistency
        self.data.set_poindat(self.pointdat)

    #we save the point data
    def save_NNpointdat(self):
        if self.data.check_key(self.NNpts_key):
            self.data[self.NNpts_key][...]=self.NN_pointdat.astype(np.float32)

    def save_and_repack(self):
        self.save_status()
        print("Repacking")
        dset_path = self.data.path_from_GUI
        self.data.close()  # close
        h5utils.repack(dset_path)
        self.data = DataSet.load_dataset(dset_path)   # Todo AD: some closing and re-opening could be factorized into a method?
        print("Repacked Dataset")
        self.update()

    #when a key for a neuron is clicked the point is now annotated. rm is the remove option
    def registerpointdat(self,i_from1,coord,rm=False):
        assert self.point_data, "Not available for mask data."
        if rm:
            print("Removing neuron",i_from1,"at time",self.i)
            self.pointdat[self.i][i_from1,:]=np.nan
            self.hlab.update_single_ci(self.data,self.i,i_from1,None)
        else:
            if any(np.isnan(self.pointdat[self.i][i_from1])):
                add = True
            else:
                add = False
            print("Setting neuron",i_from1,"at time",self.i,"to",coord)
            self.pointdat[self.i][i_from1]=coord
            self.hlab.update_single_ci(self.data,self.i,i_from1,coord)

        for client in self.calcium_registered_clients:
            client.change_ca_activity(i_from1, self.hlab.ci_int[i_from1-1][:, :2])
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
        return self.hlab.ci_int[neuron_id_from1-1][:, :2]

    def present_neurons_at_t(self, t):
        existing_annotations = np.logical_not(np.isnan(self.NN_or_GT[t][:, 0]))  # TODO AD what about masks??
        return np.nonzero(existing_annotations)[0]

    def get_seg_params(self):
        return self.data.seg_params

    def get_cluster_params(self):
        return self.data.cluster_params

    #this makes the highlight tracks
    def highlighted_track_data(self,highlighted_i_from1):
        # Todo: available for mask data??
        if not self.point_data:
            return np.array([0, 0])   # Todo: good value to return??
        trax=[]
        tray=[]
        for i in range(max(0,self.i+self.tr_pst),min(self.frame_num,self.i+self.tr_fut+1)):
            existing_neurons = np.logical_not(np.isnan(self.NN_or_GT[i][:,0]))
            if existing_neurons[highlighted_i_from1]:
                pt=self.NN_or_GT[i][highlighted_i_from1]
                trax.append(pt[0])
                tray.append(pt[1])
        return np.array([trax,tray])
