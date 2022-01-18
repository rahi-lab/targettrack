import warnings
import numpy as np
import pandas as pd


class FeatureData:
    def set_file(self, stem_savefile = None):
        """
        Creates full savefile name from stem name and stores it in self.savefile.
        :param stem_savefile: stem name for the save file ("features" and extension will be appended)
        """
        self.savefile = stem_savefile + "_features.csv"
        try:
            self.features = pd.read_csv(self.savefile)
        except FileNotFoundError:
            self.features = None

    def __init__(self, data=None, stem_savefile=None):
        self.data = data
        self.set_file(stem_savefile)

    def to_file(self):
        """Saves current state to file self.savefile"""
        if self.features is not None:
            self.features.to_csv(self.savefile, index=False, header=True, mode='w')

    @classmethod
    def from_file(cls, stem_savefile):
        """Creates instance with state loaded from savefile"""
        self = FeatureData(stem_savefile=stem_savefile)
        return self

    def feature_array(self, times=None, segments=None, rotation_invariant=False, segs_list=False):
        """
        Returns features as numpy array.
        :param times: which times to include in the feature array (all if None). Overriden by segments.
        :param segments: [(t1, s1), (t2, s2), ...] list of segments for which to return the features (in same order).
            Overrides times if given; all segments in given times if None.
        :param rotation_invariant: if True, use only rotation invariant parameters
        :param segs_list: whether to also return list of corresponding (t,s)
        :return ftrs[, segs]: ftrs the numpy array of features (one line per (time, segment));
            no columns for Time and Segment in array.
            Optionally also segs, the list of (t,s) segment corresponding to each line in ftrs, if segs_list
        """
        features = self.features
        if times is not None:
            features = self.features[self.features['Time'].isin(times)]
        if segments is not None:
            # filter for segments
            features = self.features[self.features[["Time", "Segment"]].apply(tuple, 1).isin(segments)]

        segs = features[['Time', 'Segment']].values.astype(int)
        segs = [tuple(seg) for seg in segs]

        if "Orig_cluster" in self.features.columns:
            features = features.drop(["Time", "Segment", "Orig_cluster"], axis=1)
        else:
            features = features.drop(["Time", "Segment"], axis=1)

        if rotation_invariant == 2:
            features = features[["Red Total Intensity", "Red Intensity Var.", "Red Max. Intensity", "Volume", "elongation",
                       "Rot. Inv. x loc", "Rot. Inv. y loc", "Rot. Inv. z loc", "Rot. Inv. Weighted Ixx",
                       "Rot. Inv. Weighted Iyy", "Rot. Inv. Weighted Izz", "Rot. Inv. Weighted Ixy",
                       "Rot. Inv. Weighted Ixz", "Rot. Inv. Weighted Iyz"]]
        elif rotation_invariant:
            features = features[["Red Total Intensity", "Red Intensity Var.", "Red Max. Intensity", "Volume", "elongation"]]
        # self.logger.info("Clustering based on features: {}".format(ftr.columns))

        # Sanity check in case where the feature column becomes sparse (very few objects or variation present)
        for feature in features:
            if np.sum(np.isnan(features[feature])) > 0.5 * len(features[feature]):
                warnings.warn("Sparse feature encountered, set to 0")
                features[feature] = np.zeros_like(features[feature])

        ftr_arr = np.array(features)
        if segs_list:
            return ftr_arr,segs
        else:
            return ftr_arr

    def feature_times(self):
        """
        Time for the feature array.
        :return: 1D numpy array with times corresponding to each line of the feature array:
            if line i of self.feature_array() corresponds to segment (t,s), then self.feature_times()[i] = t
        """
        return  self.features['Time'].to_numpy()

    def all_times(self):
        """All times with existing features in self."""
        return self.features['Time'].unique()

    def save_features(self, t, s, ftr_dict):
        """
        Saves given features for given segment.
        :param t: time
        :param s: segment
        :param ftr_dict: features for given (t,s); output of ph.calculate_features
        """
        # TODO
        ftr_dict.update({"Time": t, "Segment": s})
        df = pd.DataFrame([ftr_dict])
        if self.features is None:
            self.features = df
        else:
            self.features = self.features.append(df, ignore_index=True, sort=False)
            self.features.drop_duplicates(subset=("Time", "Segment"), keep="last", inplace=True)
