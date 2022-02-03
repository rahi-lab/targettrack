from src.datasets_code.DataSet import DataSet
from .AnnotationData import AnnotationData
from .worm_reader import WormReader
from src.datasets_code.EPFL_datasets.feature_data import FeatureData
from .orig_frame_info import OrigFrameInfo
from src.parameters.parameters import ParameterInitializer
import os


class nd2Data(AnnotationData, WormReader, FeatureData, OrigFrameInfo, DataSet):
    """
    dataset_path should be the path to a folder containing:
    - movie.txt, a one-liner containing the absolute path to the movie
    - all pre-existing annotations (features, segments, etc)
    """

    def available_NNpointdats(self):
        pass

    def pull_NN_results(self, NetName, runname, newpath):
        pass

    def __init__(self, dataset_path=None,
                 annotations=None, movie=None, features=None, orig_frame_info=None):
        DataSet.__init__(self, dataset_path)
        self.path_from_GUI = dataset_path
        if movie is None:
            with open(os.path.join(dataset_path, "movie.txt"), "r") as f:
                movie_file = f.readline().strip("\n")   # Todo: might have problems with \ufeff (utf8 encoding)
            WormReader.__init__(self, movie_file)
        else:
            WormReader.__init__(self, movie.movie_file)
            self.__dict__.update(movie.__dict__)
        worm_name = os.path.splitext(os.path.basename(self.movie_file))[0]
        self.name = worm_name
        stemfile = os.path.join(dataset_path, worm_name)
        if annotations is None:
            AnnotationData.__init__(self, stemfile, frame_shape=self.frame_shape)
        else:
            AnnotationData.__init__(self, "dummy")
            self.__dict__.update(annotations.__dict__)
        if features is None:
            FeatureData.__init__(self, stem_savefile=stemfile)
        else:
            FeatureData.__init__(self, stem_savefile=stemfile)
            self.__dict__.update(features.__dict__)
        if orig_frame_info is None:
            OrigFrameInfo.__init__(self, stemfile)
        else:
            OrigFrameInfo.__init__(self, stemfile)
            self.__dict__.update(orig_frame_info.__dict__)

        self.nb_channels = 2   # I think our nd2 data always has only 2 channels, red and green

        self.seg_params = ParameterInitializer.load_parameters("segmentation", stemfile)
        self.cluster_params = ParameterInitializer.load_parameters("cluster", stemfile)
        # TODO: load and save parameters
        self.point_data = False
        self._ci_int = None   # Todo: load from file if existing
        self.h5raw_filename = None


    def close(self):
        """
        Saves all data to files.
        # TODO: what's the difference?
        """
        self.to_file()

    def save(self):
        self.to_file()

    def ci_int(self):
        if self._ci_int is None:
            raise KeyError
        else:
            return self._ci_int

    def set_calcium(self, ci_int):
        self._ci_int = ci_int
        # TODO: be able to save this

    def _get_frame(self, t, col="red"):
        return self.get_3d_img(t=t, c=col)

    def _get_mask(self, t=0):
        return AnnotationData.get_mask(self, t=t, force_original=True)

    def segmented_non_ground_truth(self):
        return set(self.segmented_times()) - set(self.ground_truth_frames())

    def keys(self):
        return []   # TODO

    def to_file(self):
        '''
        Save all the computation(segmentations, rotations, features) to the file
        :return:
        '''
        AnnotationData.to_file(self)
        OrigFrameInfo.to_file(self)
        FeatureData.to_file(self)

    @classmethod
    def from_file(cls,dataset_path):
        '''
        Initialize a class from the file
        :param dataset_path: the location of the file
        :return:
        '''
        self = nd2Data.__init__(dataset_path)
        return self
