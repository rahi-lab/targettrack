import pickle as pk


class OrigFrameInfo:
    """
    Class to contain the information on frames that is used for rotating / cropping / downsampling etc.
    This class makes sure the information on frames cannot change, so that they will be consistenly resized by the
    resizer object that has access to it.
    """

    def __init__(self, stem_filename):
        self.savefile = stem_filename + "_frames_info.pickle"
        try:
            self.load_file()
        except FileNotFoundError:
            self.info_dict = {}
            self.ground_truth = set()
            self.ROI_params = None
        # todo: maybe more efficient to store list of dicts instead of dict of dicts, possible if require list of time
        #  frames at __init__

    def save_ROI_params(self, xleft, xright, yleft, yright):
        self.ROI_params = (xleft, xright, yleft, yright)

    def flag_as_gt(self, frames):
        self.ground_truth.update(frames)

    def ground_truth_frames(self):
        return self.ground_truth

    def get_ROI_params(self):
        return self.ROI_params

    def get_center_and_main_axis(self, t):
        return self.info_dict[t]["center"], self.info_dict[t]["axis1"]

    def assign_center_and_main_axis(self, t, center, axis):
        if t in self.info_dict and ("center" in self.info_dict[t] or "axis1" in self.info_dict[t]):
            raise ValueError("Cannot assign center and axis because at least one of them already exists for this time.")
        d = {"center": center, "axis1": axis}
        if t not in self.info_dict:
            self.info_dict[t] = d
        else:
            self.info_dict[t].update(d)

    def assign_transformation_matrix(self, t, ref):
        d = {'transformation_matrix': ref}
        if t not in self.info_dict:
            self.info_dict[t] = d
        else:
            self.info_dict[t].update(d)

    def assign_loss_rt(self, t, ref):
        d = {'loss_RT': ref}
        if t not in self.info_dict:
            self.info_dict[t] = d
        else:
            self.info_dict[t].update(d)

    def get_transformation(self, t):
        if t not in self.info_dict:
            return None
        elif 'transformation_matrix' in self.info_dict[t].keys():
            return self.info_dict[t]['transformation_matrix']
        else:
            return None

    def get_transformation_keys(self):
        times = self.info_dict.keys()
        a = []
        for time in times:
            if 'transformation_matrix' in self.info_dict[time].keys():
                a.append(time)
        return a

    def save_ref(self, t, ref):
        d = {'rotation_ref': ref}
        if t not in self.info_dict:
            self.info_dict[t] = d
        else:
            self.info_dict[t].update(d)

    def get_ref_frame(self, t):
        if t not in self.info_dict:
            return None
        elif 'rotation_ref' in self.info_dict[t].keys():
            return self.info_dict[t]['rotation_ref']
        else:
            return None

    def base_ref_frame(self):
        """
        Finds the original_reference for the Register_Rotate class, i.e. the reference frame against which all frames
        are aligned.
        """
        for t in self.info_dict:
            ref = self.get_ref_frame(t)
            if ref == t:
                return ref

    def ref_frames(self):
        """
        Finds all frames previously used as references by the Register_Rotate class.
        """
        ref_frames = set()
        for t in self.info_dict:
            ref = self.get_ref_frame(t)
            ref_frames.add(ref)
        ref_frames.discard(None)
        return ref_frames

    def save_score(self, t, score):
        d = {'rotation_score': score}
        if t not in self.info_dict:
            self.info_dict[t] = d
        else:
            self.info_dict[t].update(d)

    def get_score(self, t):
        return self.info_dict[t]["rotation_score"]

    def assign_isimproper(self, t, improper=0):
        d = {'rotation_improper': improper}
        if t not in self.info_dict:
            self.info_dict[t] = d
        else:
            self.info_dict[t].update(d)

    def to_file(self, filename=None):
        if filename is None:
            filename = self.savefile
        with open(filename, "wb") as f:
            pk.dump((self.info_dict, self.ground_truth, self.ROI_params), f)

    def load_file(self, filename=None):
        if filename is None:
            filename = self.savefile
        with open(filename, "rb") as f:
            self.info_dict, self.ground_truth, self.ROI_params = pk.load(f)
        return self
