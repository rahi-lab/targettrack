import pandas as pd
import sparse
import numpy as np

class AnnotationData:
    """
    Contains all the segmentation and assignment data
    WARNING: self.assignments['Clusternames'] will contain neurite ids (as strings) rather than names
    """
    # Todo: if we can preserve segments instead of merging them when two segs are one same neuron, that would help
    #  (make possible) the classification
    # TODO: what happens to features when neurons/segs are reassigned? features go rotten because the segment key is unchanged

    def __init__(self, stem_savefile, frame_shape: tuple = (512, 512, 35)):   # Todo: is it right to have a default value here?
        """
        Initialize the class for segments and assignments
        :param stem_savefile: The stem name for the files in which to save assignments and segments
        :param frame_shape: the shape of the numpy array of any frame of the video
        """
        self._normal_seg_file = stem_savefile + "_segmented.csv"
        self._coarse_seg_file = stem_savefile + "_highthresh_segmented.csv"
        self.assignment_file = stem_savefile + "_assignment.csv"
        try:
            self._normal_data_frame = pd.read_csv(self._normal_seg_file)
        except FileNotFoundError:
            self._normal_data_frame = pd.DataFrame({"Time": [], "Segment": [], "x": [], "y": [], "z": []}, dtype=int)
        try:
            self._coarse_data_frame = pd.read_csv(self._coarse_seg_file)
        except FileNotFoundError:
            self._coarse_data_frame = pd.DataFrame({"Time": [], "Segment": [], "x": [], "y": [], "z": []}, dtype=int)

        # whether to deal with coarse segmentation:
        self.coarse_seg_mode = False

        self.shape = frame_shape
        try:
            self.assignments = pd.read_csv(self.assignment_file)
        except FileNotFoundError:
            self.assignments = pd.DataFrame({"Time": [], "Segment": [], "Clusternames": []}, dtype=int)
        self.new_format()

    @property
    def data_frame(self):
        if self.coarse_seg_mode:
            return self._coarse_data_frame
        else:
            return self._normal_data_frame

    @data_frame.setter
    def data_frame(self, value):
        if self.coarse_seg_mode:
            self._coarse_data_frame = value
        else:
            self._normal_data_frame = value

    @property
    def seg_file(self):
        if self.coarse_seg_mode:
            return self._coarse_seg_file
        else:
            return self._normal_seg_file

    def new_format(self):
        """
        This is for backwards compatibility. Clusternames used to be names, now we want them to be int identifiers.
        Here we map all clusternames to unique ints.
        """
        if not len(self.assignments):
            return
        if isinstance(self.assignments["Clusternames"][0], str):   # no need to convert if they are already ints
            d = {name: i+1 for i, name in enumerate(self.assignments["Clusternames"].unique())}
            self.assignments["Clusternames"] = self.assignments["Clusternames"].map(d)
        self.assignments = self.assignments.astype(int)
        # Todo: more generally, constrain all values to int

    def segmented_times(self):
        """
        :return [t1, t2, t3, ...]: all times in this database
        """
        return self.data_frame['Time'].unique()

    def segmented_frame(self, t, coarse=None):
        """
        Gets the segmented frame.
        :param t: time frame
        :param coarse: whether to return the coarse segmentation (if not provided, depends on current mode)
        :return segmented: 3D numpy array with segmented[x,y,z] = segment_id, or 0 for background
        """
        if coarse is True:
            df = self._coarse_data_frame
        elif coarse is False:  # should not happen, but just to be complete
            df = self._normal_data_frame
        else:
            df = self.data_frame
        if t not in df.Time:
            raise KeyError
        segment_time = df[df['Time'] == t]
        frame = sparse.COO([segment_time.x, segment_time.y, segment_time.z], segment_time.Segment, shape=self.shape)
        return frame.todense()

    def get_segs_and_assignments(self, times):
        """
        Returns list of all segments for given times and corresponding list of assigned neurites.
        :param times: iterable of time frames for which to get segments
        :return segments, neurites: segments = [(t1, s1), (t1, s2), ..., (t2, s), ...] list of segments for given frames
                                    neurites = [n1, n2, ...] neurite assigned to each segment in segments
        """
        assignments = self.assignments[self.assignments['Time'].isin(times)]
        segments = assignments[['Time', 'Segment']].values
        neurites = assignments['Clusternames'].to_list()
        return segments, neurites

    def get_mask(self, t=0, force_original=False):
        """
        Gets the mask of neurons in time frame t.
        :param t: time frame
        :param force_original: True forces non-coarse segmentation
        :return: mask (3D array of same shape as frame) with neuron id for pixel value (0 for background)
        """
        if force_original:
            df = self._normal_data_frame
        else:
            df = self.data_frame
        if t not in df.Time.values or t not in self.assignments.Time.values:   # Todo: make sure t not in self.assignments.Time is not needed, as that should never happen
            raise KeyError
        segs, neus = self.get_segs_and_assignments([t])
        d = {seg[1]: neu for seg, neu in zip(segs, neus)}
        d[0] = 0
        segment_time = df[df['Time'] == t]
        frame = sparse.COO([segment_time.x, segment_time.y, segment_time.z], segment_time.Segment.map(d), shape=self.shape)
        return frame.todense()

    @property
    def real_neurites(self):
        """List of neurites of interest (not background or noise)"""
        clusternames = self.assignments['Clusternames'].unique().tolist()
        # clusternames = [cln for cln in clusternames if cln.lower() != "noise"]
        # Todo
        return clusternames

    @property
    def nb_neurons(self):
        return len(self.real_neurites)

    @nb_neurons.setter
    def nb_neurons(self, value):
        # TODO
        pass
    # TODO: get rid of real_neurites?

    # editing the data
    def _save_mask(self, t, mask):
        d = {(t, i): i for i in np.unique(mask) if i}
        self.add_segmentation(t, mask)
        self.assign(d)

    def add_segmentation(self, t, segmented):
        """
        Stores (or replaces if existing?) the segmentation for time t.
        :param t: time frame
        :param segmented: segmented frame (3D numpy array with segmented[x,y,z] = segment (0 if background)
        """
        # Todo: maybe save to savefile sometimes
        x, y, z = np.nonzero(segmented)
        s = segmented[x, y, z]
        df = pd.DataFrame({"Time": t, "Segment": s, "x": x, "y": y, "z": z})
        self.data_frame = self.data_frame.append(df, ignore_index=True, sort=False)
        self.data_frame.drop_duplicates(subset=("Time", "x", "y", "z"), keep="last", inplace=True)   # todo: can we ever duplicate values??

    def assign(self, assignment_dict, update_nb_neurons=False):
        if update_nb_neurons:   # TODO
            pass
            # nb_neurons = max(assignment_dict.values())
            # self.nb_neurons = nb_neurons
        times, segs, cls = [], [], []
        for key, val in assignment_dict.items():
            times.append(key[0])
            segs.append(key[1])
            cls.append(val)
        df = pd.DataFrame({"Time": times, "Segment": segs, "Clusternames": cls})
        self.assignments = self.assignments.append(df, ignore_index=True, sort=False)
        self.assignments.drop_duplicates(keep="last", inplace=True, subset=["Time", "Segment"])

    def to_file(self):
        self._normal_data_frame.to_csv(self._normal_seg_file, index=False, header=True, mode='w')
        self._coarse_data_frame.to_csv(self._coarse_seg_file, index=False, header=True, mode='w')
        self.assignments.to_csv(self.assignment_file, index=False, header=True, mode='w')

    @classmethod
    def from_file(cls, stem_savefile):
        """Creates instance with state loaded from savefile"""
        self = AnnotationData(stem_savefile=stem_savefile)
        return self
