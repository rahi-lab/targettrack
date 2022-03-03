from .DataSet import DataSet
from src.parameters.parameters import ParameterInitializer
import h5py
import numpy as np
import os
import warnings
from collections import defaultdict


class h5Data(DataSet):
    def __init__(self, dataset_path=None):
        self.dataset = h5py.File(dataset_path, "r+")
        super(h5Data, self).__init__(dataset_path)
        self.dataset.attrs["dset_path_from_GUI"] = dataset_path
        if "name" not in self.dataset.attrs:
            self.dataset.attrs["name"] = os.path.splitext(os.path.basename(dataset_path))[0]   # the filename without the extension
        if "seg_params" not in self.dataset:
            self.dataset.create_group("seg_params")
            for k, v in ParameterInitializer.new_parameters("segmentation", "dummy").items():
                self.dataset["seg_params"].attrs[k] = v
        if "cluster_params" not in self.dataset:
            self.dataset.create_group("cluster_params")
            for k, v in ParameterInitializer.new_parameters("clustering", "dummy").items():
                if v is None:
                    v = "None"
                self.dataset["cluster_params"].attrs[k] = v
        if "pointdat" in self.dataset:
            self.point_data = True
        if "C" not in self.dataset.attrs:   # or self.dataset.attrs["C"] is None
            self.dataset.attrs["C"] = self.dataset["0/frame"].shape[0]   # number of channels

    def __getitem__(self,key):
        return self.dataset[key]

    @property
    def point_data(self):
        if "pointdat" in self.dataset.attrs:
            return self.dataset.attrs["pointdat"]
        else:
            return None

    @point_data.setter
    def point_data(self, value: bool):
        if value is None:
            return
        self.dataset.attrs["pointdat"] = value
        if value:
            if "pointdat" not in self.dataset:
                shape = (len(self.frames), self.nb_neurons+1, 3)
                self.dataset.create_dataset("pointdat", shape, dtype="f4")
                self.dataset["pointdat"][...] = np.full(shape, np.nan, dtype=np.float32)

    @property
    def seg_params(self):
        '''
        Returns segmentation parameters
        :return: segmentation parameters
        '''
        return self.dataset["seg_params"].attrs

    @property
    def cluster_params(self):
        '''
        Returns cluster parameters
        :return: cluster parameters
        '''
        return self.dataset["cluster_params"].attrs

    @property
    def name(self):
        '''
        Returns the name of the experiment or run, i.e name of the h5 file to be precise
        :return: string
        '''
        return self.dataset.attrs["name"]

    @property
    def path_from_GUI(self):
        '''
        :returns the dataset path
        :return: string
        '''
        return self.dataset.attrs["dset_path_from_GUI"]

    @property
    def nb_channels(self):
        '''
        Returns number of channels of the image data.
        :return: Integer
        '''
        return self.dataset.attrs["C"]

    @property
    def frame_shape(self):
        '''
        Returns the dimension of the frame
        :return:
        '''
        return self.dataset.attrs["W"], self.dataset.attrs["H"], self.dataset.attrs["D"]

    @property
    def nb_neurons(self):
        '''
        Return the list of the  neurons
        :return: list
        '''
        return self.dataset.attrs["N_neurons"]

    @nb_neurons.setter
    def nb_neurons(self, value):
        self.dataset.attrs["N_neurons"] = value

    @property
    def h5raw_filename(self):
        try:
            return self.dataset["h5rawfn"]
        except KeyError:
            return None

    @property
    def pointdat(self):
        """
        pointdat is a self.frame_num * (self.nb_neurons+1) * 3 array with:
        pointdat[t][n] = [x,y,z] where x,y,z are the coordinates of neuron n in time frame t (neurons start
        at n>=1, 0 is for background and contains np.nans)
        """
        return np.array(self.dataset["pointdat"])

    @property
    def frame_num(self):
        return self.dataset.attrs["T"]

    @property
    def frames(self):
        return list(range(self.dataset.attrs["T"]))

    @property
    def real_neurites(self):
        return list(range(1, self.nb_neurons+1))   # Todo: nb_neurons must be up-to-date

    def close(self):
        self.dataset.close()

    def save(self):
        pass

    def segmented_times(self):
        # Todo: for Harvard lab data, should it filter and return only frames with pointdat?
        if self.coarse_seg_mode:
            mask_key = "coarse_mask"
        else:
            mask_key = "mask"
        return [t for t in self.frames if str(t) + "/{}".format(mask_key) in self.dataset]

    def ground_truth_frames(self):
        if "ground_truth" not in self.dataset.attrs:
            self.dataset.attrs["ground_truth"] = []
        return list(self.dataset.attrs["ground_truth"])

    def segmented_non_ground_truth(self):#MB defined this for classification
        nonground = set(self.segmented_times())-set(self.ground_truth_frames())
        return nonground

    def get_transformation_keys(self):
        return [t for t in self.frames if "{}/transfo_matrix".format(t) in self.dataset]

    ####################################################################################
    # reading the data

    def __getitem__(self,key):
        return self.dataset[key]

    def check_key(self, key):
        return key in self.dataset

    def _get_frame(self, t, col="red"):
        '''
        The frame
        :param t: Integer, the time
        '''
        return np.array(self.dataset[str(t) + "/frame"])[0 if col == "red" else 1]

    def _get_mask(self, t):
        '''
        Get the mask of neurons in frame t.
        :param t: Integer, the time
        '''
        if self.coarse_seg_mode:
            mask_key = "coarse_mask"
        else:
            mask_key = "mask"
        return np.array(self.dataset[str(t) + "/{}".format((mask_key))])

    def get_validation_set(self,NNnames):
        """
        MB : this is defined to get the ids of the frames that were used as
        validation set in the NNname neural network
        """
        Validation = set()
        if 'Validinds' in  self.dataset['net'][NNnames].attrs.keys():
            Validation = self.dataset['net'][NNnames].attrs['Validinds']
            Validation = np.sort(Validation)
        return Validation

    def segmented_frame(self, t, coarse=None):
        """
        Gets the segmented frame.
        :param t: time frame
        :param coarse: whether to return the coarse segmentation (if not provided, depends on current mode)
        :return segmented: 3D numpy array with segmented[x,y,z] = segment_id, or 0 for background
        """
        if coarse is True:
            seg_key = "coarse_seg"
        elif coarse is False:  # should not happen, but just to be complete
            seg_key = "seg"
        else:
            if self.coarse_seg_mode:
                seg_key = "coarse_seg"
            else:
                seg_key = "seg"
        skey = str(t) + "/{}".format(seg_key)
        if not skey in self.dataset.keys():
            print("segmentation not found for this frame ")#MB added to extract the troubling frame
            print(t)
        return np.array(self.dataset[skey])

    def feature_array(self, times=None, segments=None, rotation_invariant=False, segs_list=False, further_alignments=False):
        """
        Returns features as numpy array (to be used for clustering/classification).
        :param times: which times to include in the feature array (all if None). Overriden by segments.
        :param segments: [(t1, s1), (t2, s2), ...] list of segments for which to return the features (in same order).
            Overrides times if given; all segments in given times if None.
        :param rotation_invariant: if True, use only rotation invariant parameters
        :param further_alignment: uses all the ffeatures for clustering (both aligned and not aligned features)
        :param segs_list: whether to also return list of corresponding (t,s)
        :return ftrs[, segs]: ftrs the numpy array of features (one line per (time, segment));
            no columns for Time and Segment in array.
            Optionally also segs, the list of (t,s) segment corresponding to each line in ftrs, if segs_list
        """
        ExtractFromSeg=self.use_seg_for_feature
        if times is not None and ExtractFromSeg:
            print("get labels from seg")
            segs = [(t, s) for t in times for s in np.unique(self.segmented_frame(t))[1:]]
        elif times is not None and not ExtractFromSeg:
            print("get labels from mask")
            segs = [(t, s) for t in times for s in np.unique(self.get_mask(t))[1:]]#if you are using masks and masks are different from segs

        if segments is not None:
            segs = segments

        if rotation_invariant == 2:
            raise NotImplementedError
        elif rotation_invariant:
            filt = [1, 2, 3, 0, 10]
        elif further_alignments:
            filt = ...
        else:
            filt = range(11)  # todo: warning, the order may not be the same as the nd2 version
        print('id features: '+str(filt))

        print(segs)
        features = np.vstack([np.array(self.dataset["features/{}/{}".format(t, s)])[filt] for t, s in segs])
        # Sanity check in case where the feature column becomes sparse (very few objects or variation present)
        feature_is_sparse = np.sum(np.isnan(features), axis=0) > (0.5 * features.shape[0])
        if any(feature_is_sparse):
            warnings.warn("{} sparse feature encountered and set to 0".format(np.sum(feature_is_sparse)))
            features[:, feature_is_sparse] = 0

        if segs_list:
            return features, segs
        else:
            return features

    def get_segs_and_assignments(self, times):
        '''MB's understanding: for each time frame the segmentation of that fram
        and the mask corresponding to it (from clustering) is recovered. non-Noise
        segments in each time frames are extracted and the pairs (t,s) Stores
        each frame and segments present in it. for each of this present segments,
        one checks the assigned mask to it (neuron number I guess). to get the assigned mask, one pixel with the mentioned
        segment value is traced in the mask matrix. this value in the mask matrix(neu)
        is assigned to the (t,s doublet)

        Retuns: ordered doublets of (t,s) and the neuron value assigned to each of them
        '''
        # Todo: a bit more efficient...
        segs = []
        neus = []
        print("Getting segments and assignments for frames:")#MB added
        for t in times:
            print(t)
            t = int(t)#MB added
            segmented = self.segmented_frame(t)
            mask = self.get_mask(t)
            xs, ys, zs = np.nonzero(segmented)#coordinates of non-noise segments
            this_segs = np.unique(segmented[xs, ys, zs])
            segs.extend([(t,s) for s in this_segs])#segments present in each t
            for s in this_segs:
                id = np.argwhere(segmented[xs, ys, zs] == s)[0]#first triplet that gives the coordinate of 1st pixel of segment s
                #neu = mask[xs, ys, zs][id[0], id[1], id[2]]#mask id of each present segment in frame t
                neu = mask[xs[id], ys[id], zs[id]]#MB version,mask id of each present segment in frame t
                neus.append(neu)
        return segs, neus


    def get_transformation(self, t):
        key = "{}/transfo_matrix".format(t)
        if key in self.dataset:
            return np.array(self.dataset[key])
        return None

    def ref_frames(self):
        if "ref_frames" not in self.dataset:
            return set()
        ref_frames = set(self.dataset["ref_frames"].attrs.values())
        return ref_frames

    def base_ref_frame(self):
        """
        Finds the original_reference for the Register_Rotate class, i.e. the reference frame against which all frames
        are aligned.
        """
        if "ref_frames" not in self.dataset:
            return
        for t, ref in self.dataset["ref_frames"].attrs.items():
            if ref == int(t):
                return ref

    def get_score(self, t):
        return self.dataset["rotation_score"].attrs[str(t)]

    def get_ROI_params(self):
        return self.dataset.attrs["ROI"]

    def available_NNpointdats(self):
        if "net" not in self.dataset:
            return []
        return self.dataset["net"].keys()

    def ci_int(self):
        return self.dataset["ci_int"][:,:,:]

    ####################################################################################
    # editing the data

    def replace_frame(self, t, img_red, img_green):
        old_img = np.array(self.dataset[str(t) + "/frame"], dtype=np.int16)
        self.dataset[str(t) + "/frame"][...] = np.stack([img_red, img_green]).astype(np.int16)
        orig_key = str(t) + "/oriframe"
        if orig_key not in self.dataset:
            self.dataset.create_dataset(orig_key, old_img.shape, dtype="i2", compression="gzip")
        self.dataset[orig_key][...] = old_img

    def _save_frame(self, t, frameR,frameG=0, mask=0):#MB added
        '''
        saves the cropped and aligned frames with their masks in a separate h5 file for further use
        :param t: Integer, the time
        :param mask: nparray, the mask, 0 if there is no mask to be Saved
        :param frameR/G: Red channel and green channel frames if available
        :return:
        '''
        SizeR = np.shape(frameR)
        if np.any(frameG):
            frameTot = np.zeros((2,SizeR[0],SizeR[1],SizeR[2]))
        else:
            frameTot = np.zeros((1,SizeR[0],SizeR[1],SizeR[2]))
        if self.point_data is None:
            self.point_data = False
        elif self.point_data:
            raise ValueError("Masks and point data would interfere.")
        #frameTot[0,:,:,:,] = frameR
        #frameTot[1,:,:,:] = frameG
        frame_key = "frame"
        fkey = str(t) + "/{}".format(frame_key)

        if np.any(frameG):
            frameTot[0] = frameR
            frameTot[1] = frameG
        else:
            frameTot[0] = frameR
        if fkey not in self.dataset:
            self.dataset.create_dataset(fkey, frameTot.shape, dtype="i2", compression="gzip")
        else:
            del self.dataset[fkey]
            self.dataset.create_dataset(fkey, frameTot.shape,  dtype="i2", compression="gzip")
        self.dataset[fkey][...] = frameTot.astype(np.int16)
        self.dataset.attrs["W"] = SizeR[0]
        self.dataset.attrs["H"] = SizeR[1]
        self.dataset.attrs["D"] = SizeR[2]


        if np.any(mask):
            if self.coarse_seg_mode:
                mask_key = "coarse_mask"
                seg_key = "coarse_seg"
            else:
                mask_key = "mask"
                seg_key = "seg"
                mkey = str(t) + "/{}".format(mask_key)
                skey = str(t) + "/{}".format(seg_key)
            for key in (mkey, skey):
                if key in self.dataset:
                    del self.dataset[key]
                self.dataset.create_dataset(key, mask.shape,  dtype="i2", compression="gzip")
                self.dataset[key][...] = mask.astype(np.int16)

    def _save_mask(self, t, mask):
        '''
        Assign the mask. MB: I think this is used only in segmentation, so after segmentation the result is saved both in mask and seg dataset
        :param t: Integer, the time
        :param mask: nparray, the mask
        :return:
        '''
        if self.point_data is None:
            self.point_data = False
        elif self.point_data:
            raise ValueError("Masks and point data would interfere.")
        if self.coarse_seg_mode:
            mask_key = "coarse_mask"
            seg_key = "coarse_seg"
        else:
            mask_key = "mask"
            seg_key = "seg"
        mkey = str(t) + "/{}".format(mask_key)
        skey = str(t) + "/{}".format(seg_key)
        for key in (mkey, skey):
            if key not in self.dataset:
                self.dataset.create_dataset(key, mask.shape, dtype="i2", compression="gzip")
            self.dataset[key][...] = mask.astype(np.int16)

    def _save_green_mask(self, t, mask):
        '''
         MB: Assigns the green mask. if there are no mask from the red channel it saves the whole mask.
         if there is already a red channel mask, it only annotates the parts that are not
         labeled by the red mask
        :param t: Integer, the time-id of the frame that the mask corresponds to
        :param mask: nparray, the mask
        :return:
        '''
        if self.point_data is None:
            self.point_data = False
        elif self.point_data:
            raise ValueError("Masks and point data would interfere.")
        if self.coarse_seg_mode:
            mask_key = "coarse_mask"
            seg_key = "coarse_seg"
        else:
            mask_key = "mask"
            seg_key = "seg"
        mkey = str(t) + "/{}".format(mask_key)
        skey = str(t) + "/{}".format(seg_key)
        for key in (mkey, skey):
            if key not in self.dataset:
                self.dataset.create_dataset(key, mask.shape, dtype="i2", compression="gzip")
                self.dataset[key][...] = mask.astype(np.int16)
            else:
                maskRed = self.dataset[key][...]
                maskRed[maskRed==0] = mask[maskRed==0]
                self.dataset[key][...] = maskRed.astype(np.int16)


    def flag_as_gt(self, frames):
        if "ground_truth" not in self.dataset.attrs:
            self.dataset.attrs["ground_truth"] = list(frames)
        else:
            self.dataset.attrs["ground_truth"] = np.hstack([self.dataset.attrs["ground_truth"], frames])

    def save_features(self, t, s, ftr_dict):
        key = "features/{}/{}".format(t,s)
        list = self._feature_dict_to_array(ftr_dict)
        if key in self.dataset:
            del self.dataset[key]
        self.dataset.create_dataset(key, (len(list),),dtype="float", compression="gzip")   # Todo: choose dtype?
        self.dataset[key][...] = list

    @classmethod
    def _feature_dict_to_array(cls, ftr_dict):
        # Todo: there may be more keys in this dict
        KEY = ftr_dict.keys()
        lst = [ftr_dict[lab] for lab in KEY#["Volume", "Red Total Intensity", "Red Intensity Var.", "Red Max. Intensity",
                                         #"Weighted Ixx", "Weighted Iyy", "Weighted Izz", "Weighted Ixy", "Weighted Ixz",
                                         #"Weighted Iyz", "elongation"]
               ]
        return lst

    def assign(self, assignment_dict, update_nb_neurons=False):
        if update_nb_neurons:
            nb_neurons = max(assignment_dict.values())
            self.nb_neurons = nb_neurons
        time_dict = defaultdict(dict)
        for (t,s), n in assignment_dict.items():
            time_dict[t][s] = n
        for t, dct in time_dict.items():
            print(t)
            print(dct)
            MaxId= np.max(list(dct.values()))+1
            print('Maxs:'+str(MaxId))
            # this is the function that assigns a neurite to a segment of time t
            def fun(x):
                return dct[x] if x else 0#changed 0 to MaxId
            np_fun = np.vectorize(fun)
            # this is the segmentation:
            seg_mask = self.segmented_frame(t)

            # we suppose that frames are sparse, therefore it is faster to apply the assignment to nonzero values only
            #TODO if a segment is assigned to none of the neurons raise an error --MB added
            xs, ys, zs = np.nonzero(seg_mask)
            vals = seg_mask[xs, ys, zs]
            mask_vals = np_fun(vals)
            mask_vals[mask_vals==0] = MaxId#MB addded to avoid setting values to 0
            mask = np.zeros_like(seg_mask)
            mask[xs, ys, zs] = mask_vals

            # now save computed mask
            if self.coarse_seg_mode:
                mask_key = "coarse_mask"
            else:
                mask_key = "mask"
            key = str(t) + "/{}".format(mask_key)
            self.dataset[key][...] = mask.astype(np.int16)

    def save_transformation_matrix(self, t, matrix):
        key = "{}/transfo_matrix".format(t)
        if key not in self.dataset:
            self.dataset.create_dataset(key, data=matrix)
        else:
            self.dataset[key][...] = matrix

    def save_ref(self, t, ref):
        if "ref_frames" not in self.dataset:
            self.dataset.create_group("ref_frames")
        self.dataset["ref_frames"].attrs[str(t)] = ref

    def save_score(self, t, score):
        if "rotation_score" not in self.dataset:
            self.dataset.create_group("rotation_score")
        self.dataset["rotation_score"].attrs[str(t)] = score

    def save_ROI_params(self, xleft, xright, yleft, yright):
        self.dataset.attrs["ROI"] = [xleft, xright, yleft, yright]

    def set_poindat(self, pointdat):
        '''
        Set the pointdat
        pointdat[t][n] = [x,y,z] where x,y,z are the coordinates of neuron n in time frame t (neurons start
        at n>=1, 0 is for background and contains np.nans)
        :param pointdat: The value
        :return:
        '''
        if self.point_data is None:
            self.point_data = True
        elif not self.point_data:
            raise ValueError("Masks and point data would interfere.")

        self.dataset["pointdat"][...] = pointdat.astype(np.float32)
        # TODO: deal with changing nb of neurons

    def set_NN_pointdat(self, key):
        '''
        set NN pointdat
        :param key: key
        :return:
        '''
        assert self.point_data, "Masks and point data would interfere."
        nn_dset = self.dataset["net/" + key + "/NN_pointdat"]
        if "NN_pointdat" not in self.dataset:
            self.dataset.create_dataset("NN_pointdat", nn_dset.shape, dtype="f4")
        self.dataset["NN_pointdat"][...] = nn_dset

    # TODO: make sure results are correctly accessed, including for masks.
    def pull_NN_results(self, NetName, runname, newpath):
        if self.point_data is None:
            self.point_data = True
        elif not self.point_data:
            raise ValueError("Masks and point data would interfere.")
        identifier = "net/" + NetName + "_" + runname
        del self.dataset[identifier]
        if "net" not in self.dataset.keys():
            self.dataset.create_group("net")
        h5net = h5py.File(newpath, "r")
        h5net.copy(identifier, self.dataset["net"])
        h5net.close()
        print("Merging Training results of ", NetName+"_"+runname, " into ", self.name)

    def set_calcium(self, ci_int):
        """
        this sets the ci_int data
        """
        assert (self.nb_neurons==ci_int.shape[0]) and (len(self.frames)==ci_int.shape[1]),"ci_int Shape mismatch"
        if "ci_int" in self.dataset:
            del self.dataset["ci_int"]
        self.dataset.create_dataset("ci_int", shape=ci_int.shape, dtype="f4", compression="gzip")
        self.dataset["ci_int"][...] = ci_int
