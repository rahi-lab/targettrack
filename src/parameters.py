import logging
import pickle as pk


class Parameters(dict):
    """
    Wrapper around the dict class, providing the fields:
        - kind to register which parameters are stored
        - savefile the exact filename to save/load data from
    and methods:
        - to_file and from_file to save/load dictionary in pickle format
    """
    # Todo: find a way to include help about the parameters
    def __init__(self, kind, savefile=None, *args, **kwargs):
        super(Parameters, self).__init__(*args, **kwargs)
        self.kind = kind
        self.savefile = savefile
        self.logger = logging.getLogger('Params')

    def to_file(self):
        with open(self.savefile, "wb") as f:
            pk.dump(dict(self), f)
        self.logger.debug("Saved parameters to file {}".format(self.savefile))

    @classmethod
    def from_file(cls, kind, savefile):
        self = cls(kind, savefile)
        with open(self.savefile, "rb") as f:
            self.update(pk.load(f))
        self.logger.debug("Loaded parameters from file {}".format(self.savefile))
        return self

    @classmethod
    def pyqt_param_keywords(cls, param_name):
        if param_name in {"min_dist", "minvol", "graph_nneighbors", "min_pixels_object", "large_obj_threshold",
                          "max_pixels_object"}:
            return {"type": "int", "step": 1}
        elif param_name in {"perc", "st_factor", "sigm", "sigm_dtr", "bg_factor", "dist_threshold"}:
            return {"type": "float", "step": 0.01}
        elif param_name in {"pc_var", "clrange"}:
            return {"type": "str"}
        elif param_name in {"rotation_invariant", "graph_cluster", "further_alignments"}:
            return {"type": "bool"}


class ParameterInitializer:
    """
    Class to initialize Parameters objects with appropriate default values or saved values.
    """
    @classmethod
    def load_parameters(cls, kind, stem_savefile):
        """Tries to load parameters from file, but uses default parameters if file is missing."""
        savefile = stem_savefile + "." + kind + "_params.pickle"
        try:
            params = Parameters.from_file(kind, savefile)
        except FileNotFoundError:
            params = cls.new_parameters(kind, stem_savefile)
        return params

    @classmethod
    def new_parameters(cls, kind, stem_savefile):
        """

        :param kind: defines which parameters are to be stored. This is used to determine savefile name and default values.
            Must be one of ["segmentation", "clustering", "cnn"].
        :param stem_savefile: stemname for save file to load/save parameters from.
        :return:
        """
        savefile = stem_savefile + "." + kind + "_params.pickle"
        if "seg" in kind:
            default_values = {"sigm": 1, "bg_factor": 25, "perc": 0.97, "sigm_dtr": 5, "min_dist": 1, "st_factor": 0.2,
                              "minvol": 3, "min_pixels_object": 120, "large_obj_threshold": 500, "dist_threshold": 10,
                              "max_pixels_object": 100000}
        elif "cluster" in kind:
            default_values = {"pc_var":None, "clrange":(3,40), "rotation_invariant":True,#MB changed the range from(3,20) to (3,40)
                              "further_alignments":False,"graph_cluster":False, "graph_nneighbors":20}  # Set pc_var to None until pca issue is solved
        elif "cnn" in kind.lower():
            default_values = {"prop_segmented_for_cnn":1., "display_cnn_results":False,
                              "extract_activity_from_both":False, "train_with_gt_only":False, "epochs":100,
                              "batch_size":1, "align":True}
            # Todo: update this
        else:
            raise ValueError("Unknown parameter kind.")
        params = Parameters(kind, savefile, default_values)
        return params
