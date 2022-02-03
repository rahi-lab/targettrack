import abc
import numpy as np
from src.graphic_interface.image_standardizer import ImageAligner, ImageCropper



class DataSet:
    """
    Wrapper around both Dataset, which can apply a transformation prior to sending data, or apply
    a reverse transformation after receiving data.
    """

    def __init__(self, dataset_path=None):
        self.aligner = None
        self.cropper = None
        self._align = False
        self.crop = False
        self.coarse_seg_mode = False # MB added
        self.only_NN_mask_mode = False # MB
        self.use_seg_for_feature = False #MB added

        self.point_data = None

        # list of abstract attributes, that should be defined by inheriting classes:
        # self.seg_params = None
        # self.cluster_params = None
        # self.frames = None   # the iterable of frames
        # self.frame_num = None   # the number of frames
        # self.name = None
        # self.path_from_GUI = None
        # self.nb_channels = None
        # self.frame_shape = None
        # self.nb_neurons = None
        # self.h5raw_filename = None
        # self.pointdat = None   # Todo: get rid of calls in nd2??
        # self.pointdat is a self.frame_num * (self.nb_neurons+1) * 3 array with:
        # self.pointdat[t][n] = [x,y,z] where x,y,z are the coordinates of neuron n in time frame t (neurons start
        # at n>=1, 0 is for background and contains np.nans)
        # self.NN_pointdat = None

    @classmethod
    def load_dataset(cls, dataset_path):
        if dataset_path.endswith(".nd2"):
            from .nd2Data import nd2Data
            return nd2Data(dataset_path)
        else:
            from .h5Data import h5Data
            return h5Data(dataset_path)


    @property
    def align(self):
        return self._align

    @align.setter
    def align(self, value):
        if value and self.aligner is None:
            self.aligner = ImageAligner(self)
        self._align = value

    @property
    def crop(self):
        return self._crop

    @crop.setter
    def crop(self, value):
        if value and self.cropper is None:
            orig_shape = self.get_frame(0, force_original=True).shape
            self.cropper = ImageCropper(self, orig_shape)
        self._crop = value

    @abc.abstractmethod
    def close(self):
        """Close and/or save dataset properly"""   # TODO
        raise NotImplementedError

    @abc.abstractmethod
    def save(self):
        """Save what??"""   # TODO
        raise NotImplementedError

    @abc.abstractmethod
    def segmented_times(self):
        """Gets the list of times for which a segmentation is defined"""
        raise NotImplementedError

    @abc.abstractmethod
    def ground_truth_frames(self):
        """Gets the list of frames marked as ground truth"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_transformation_keys(self):
        """Gets the list of frames for which a transformation is defined."""
        raise NotImplementedError

    ####################################################################################
    # reading the data
    @abc.abstractmethod   # TODO AD GET RID OF THIS!!! I think it's just a relic of when dataset was directly an h5
    def __getitem__(self,key):
        """returns data[key]"""
        raise NotImplementedError

    @abc.abstractmethod
    def check_key(self, key):
        """checks if key exists"""
        raise NotImplementedError

    @abc.abstractmethod
    def _get_frame(self, t, col="red"):
        """Gets the original frame"""
        raise NotImplementedError

    def get_frame(self, t, col="red", force_original=False):
        frame = self._get_frame(t, col=col)
        if force_original:
            return frame
        return self._transform(t, frame)

    @abc.abstractmethod
    def _get_mask(self, t):
        """
        Gets the original mask of neurons
        Raises KeyError if no mask exists for time t.
        """
        raise NotImplementedError


    def get_mask(self, t, force_original=False):
        """
        Gets the segmented frame.
        :param t: time frame
        :param force_original: forces to return mask corresponding to original image, without transformation, and from non-coarse segmentation (if applicable)
        :return segmented: 3D numpy array with segmented[x,y,z] = segment_id, or 0 for background
        """
        orig_segmented = self._get_mask(t)
        if force_original:
            return orig_segmented
        return self._transform(t, orig_segmented, True)

    @abc.abstractmethod
    def segmented_frame(self, t, coarse=None):
        """
        Gets the segmentation mask (NOT the neurons mask) for frame t.
        Never applies a transformation.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_validation_set(self, NNname):   # MB added this
        """
        gets the frames that are validation set in NN
        """
        raise NotImplementedError


    @abc.abstractmethod
    def feature_array(self):
        """
        Returns features as numpy array (to be used for clustering/classification).
        :param times: which times to include in the feature array (all if None). Overriden by segments.
        :param segments: [(t1, s1), (t2, s2), ...] list of segments for which to return the features (in same order).
            Overrides times if given; all segments in given times if None.
        :param rotation_invariant: if True, use only rotation invariant parameters
        :param segs_list: whether to also return list of corresponding (t,s)
        :return ftrs[, segs]: ftrs the numpy array of features (one line per (time, segment));
            no columns for Time and Segment in array.
            Optionally also segs, the list of (t,s) segment corresponding to each line in ftrs, if segs_list
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_segs_and_assignments(self, times):
        """
        Returns list of all segments for given times and corresponding list of assigned neurites.
        :param times: iterable of time frames for which to get segments
        :return segments, neurites: segments = [(t1, s1), (t1, s2), ..., (t2, s), ...] list of segments for given frames
                                    neurites = [n1, n2, ...] neurite assigned to each segment in segments
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_transformation(self, t):
        """
        Gets the matrix of affine transformation to be applied to align given frame.
        :param t: time frame
        :return: linear transformation matrix as output by Register_rotate.composite_transform
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ref_frames(self):
        """Gets the set of frames used as rotation references."""
        raise NotImplementedError

    @abc.abstractmethod
    def base_ref_frame(self):
        """
        Finds the original_reference for the Register_Rotate class, i.e. the reference frame against which all frames
        are aligned.
        """
        raise NotImplementedError

    # Todo: get_ref(t)??

    @abc.abstractmethod
    def get_score(self, t):
        """
        Gets the rotation score of frame t.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_ROI_params(self):
        """
        Gets the the boundaries of the Region Of Interest (minimum region to include when cropping)
        :return: xleft, xright, yleft, yright: the boundaries of the ROI.
        """

    @abc.abstractmethod
    def available_NNpointdats(self):
        """Gets iterable of NN ids for which pointdat is available"""
        raise NotImplementedError

    def get_existing_neurons(self, t):
        """
        :param t: time
        :return existing_neurons: boolean array of len self.nb_neurons+1 with existing_neurons[neu] is True iff
            neurite neu exists at time t
        """
        if self.point_data:
            existing_neurons = np.logical_not(np.isnan(self.pointdat[t][:, 0]))
        elif self.point_data is None:
            existing_neurons = np.full(self.nb_neurons+1, False)
        else:
            try:
                mask = self.get_mask(t)
                neurons = np.unique(mask)[0:]
                existing_neurons = np.array([False] + [n in neurons for n in range(1, self.nb_neurons + 1)])
            except KeyError:
                existing_neurons = np.full(self.nb_neurons + 1, False)
        return existing_neurons

    @abc.abstractmethod
    def ci_int(self):
        """Raise KeyError if they are not defined."""  # Todo: in what conditions?? useful to have KeyError?
        raise NotImplementedError

    ####################################################################################
    # editing the data

    @abc.abstractmethod
    def replace_frame(self, t, img_red, img_green):
        """
        Replaces the videoframe of time frame t by the provided images img_red and img_green, which should be of same
        size as original image. Stores the original image into another field, for memory.
        Only supposed to work for two channels (though that can be easily changed)
        :param t: time frame
        :param img_red: new frame (3D numpy array) for red channel
        :param img_green: new frame (3D numpy array) for red channel
        """
        raise NotImplementedError
    @abc.abstractmethod
    def _save_frame(self, t, frame):#MB added
        raise NotImplementedError

    def save_frame(self, t, frameR, frameG = 0, mask = 0, force_original=False):#MB added
        """
        Stores (or replaces if existing?) the segmentation for time t.
        :param t: time frame
        :param mask: segmented frame (3D numpy array with segmented[x,y,z] = segment (0 if background)
        :param force_original: if True, does not apply inverse transform (otherwise, respects self.crop and self.align)
        :param update_nb_neurons: whether to update self.nb_neurons to match nb of neurons in mask.
            More precisely, self.nb_neurons can only be increased to nb of neurons in mask.
            Supposes that the max neuron id in the mask is the nb of neurons.
        """
        if not force_original:
            frameR = self._reverse_transform(t, frameR)
            if np.any(frameG):
                frameG = self._reverse_transform(t, frameG)
            if mask:
                mask =  self._reverse_transform(t, mask)
        if np.any(mask):
            self._save_frame(t,frameR,frameG,mask)
        else:
            self._save_frame(t,frameR,frameG, 0)
    @abc.abstractmethod
    def _save_mask(self, t, mask):
        raise NotImplementedError

    def save_mask(self, t, mask, force_original=False, update_nb_neurons=False):
        """
        Stores (or replaces if existing?) the segmentation for time t.
        :param t: time frame
        :param mask: segmented frame (3D numpy array with segmented[x,y,z] = segment (0 if background)
        :param force_original: if True, does not apply inverse transform (otherwise, respects self.crop and self.align)
        :param update_nb_neurons: whether to update self.nb_neurons to match nb of neurons in mask.
            More precisely, self.nb_neurons can only be increased to nb of neurons in mask.
            Supposes that the max neuron id in the mask is the nb of neurons.
        """
        if update_nb_neurons:
            nb_neurons = int(mask.max()) # SJR: For some reason was getting back a float, at least if max was zero
            #if nb_neurons > self.nb_neurons: # SJR: I am not sure why this was in here, why can the number of neurons only increase? commented out
            self.nb_neurons = nb_neurons
        if not force_original:
            mask = self._reverse_transform(t, mask)
        self._save_mask(t, mask)

    def _save_green_mask(self, t, mask):
        raise NotImplementedError

    def save_green_mask(self, t, mask, force_original=False, update_nb_neurons=False):
        """
        Stores (or replaces if existing?) the segmentation for time t.
        :param t: time frame
        :param mask: segmented frame (3D numpy array with segmented[x,y,z] = segment (0 if background)
        :param force_original: if True, does not apply inverse transform (otherwise, respects self.crop and self.align)
        :param update_nb_neurons: whether to update self.nb_neurons to match nb of neurons in mask.
            More precisely, self.nb_neurons can only be increased to nb of neurons in mask.
            Supposes that the max neuron id in the mask is the nb of neurons.
        """
        if update_nb_neurons:
            nb_neurons = int(mask.max()) # SJR: For some reason was getting back a float, at least if max was zero
            #if nb_neurons > self.nb_neurons: # SJR: I am not sure why this was in here, why can the number of neurons only increase? commented out
            self.nb_neurons = nb_neurons
        if not force_original:
            mask = self._reverse_transform(t, mask)
        self._save_green_mask(t, mask)

    @abc.abstractmethod
    def flag_as_gt(self, frames):
        """Flag all given frames as ground truth"""
        raise NotImplementedError

    @abc.abstractmethod
    def save_features(self, t, s, ftr_dict):
        """Saves the given features for segment s of time t."""
        raise NotImplementedError

    @abc.abstractmethod
    def assign(self, assignment_dict, update_nb_neurons=False):
        """
        Assigns segments to neurites according to given dictionary.
        :param assignment_dict: dictionary (t, s) -> neurite
        :param update_nb_neurons: whether to update self.nb_neurons to match nb of neurons in mask.
            More precisely, self.nb_neurons can be increased or decreased to match exaclty nb of neurons in mask.
            Supposes that the max neuron id in the mask is the nb of neurons.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_transformation_matrix(self, t, matrix):
        """Saves given matrix as the matrix of affine transfo to be applied to align given frame."""
        raise NotImplementedError

    @abc.abstractmethod
    def save_ref(self, t, ref):
        """Saves that frame ref is to be used as reference for the alignment of frame t."""
        raise NotImplementedError

    @abc.abstractmethod
    def save_score(self, t, score):
        """Saves the rotation score of frame t."""
        raise NotImplementedError

    @abc.abstractmethod
    def save_ROI_params(self, xleft, xright, yleft, yright):
        """
        Saves given parameters as the boundaries of the Region Of Interest (minimum region to include when cropping)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_poindat(self, pointdat):
        raise NotImplementedError

    @abc.abstractmethod
    def set_NN_pointdat(self, key):
        '''
        set NN pointdat
        :param key: key
        :return:
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def pull_NN_results(self, NetName, runname, newpath):
        """
        Reads the NN pointdat results from h5 file named by key.
        :param key: designates an NN run from which to get the data.
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_calcium(self, ci_int):
        raise NotImplementedError

    ####################################################################################
    # defining the transformations

    def _transform(self, t, img, is_mask=False):
        if self.align:
            img = self.aligner.align(img, t, is_mask)
        if self.crop:   # TODO: crop and/or resize??
            img = self.cropper.crop(img)
        return img


    def _reverse_transform(self, t, img):
        """
        Depending on current transformation mode, applies the necessary reverse transformations to img which is assumed
        to be a mask.
        """
        if self.crop:
            img = self.cropper.inverse_crop(img)
        if self.align:
            img = self.aligner.dealign(img, t)
        return img
