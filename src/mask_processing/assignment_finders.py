from abc import abstractmethod


class AssignmentFinderFactory:
    @classmethod
    def create_assigner(cls, kind, annotations, *args, **kwargs):
        if "cluster" in kind:
            from src.mask_processing.clustering import Clustering
            assigner = Clustering(annotations, *args, **kwargs)
        elif "classifi" in kind or "clf" in kind:
            from src.mask_processing.classification import Classification
            assigner = Classification(kind, annotations, *args, **kwargs)
        else:
            raise ValueError("Unkown assigner kind. kind must contain one of ('cluster', 'clf', 'classifi').")
        return assigner


# Todo: there could be a parameter "overwrite" to some methods of the above or below classes

class AssignmentFinderInterface:
    """
    This class defines the interface for classes that implement algorithms for automatic or manual?#Todo
    assignment of segments to neurites.
    """
    def __init__(self, dataset, *args, **kwargs):
        """
        Class to find assignments for segments.
        :param dataset: instance of Dataset
        :param args, kwargs: any class-specific additional arguments
        """
        self.annotations = dataset

    @abstractmethod
    def prepare(self, *args, **kwargs):
        """
        Any preliminary actions, such as classifier training.
        """
        raise NotImplementedError

    @abstractmethod
    def find_assignment(self, frames, *args, **kwargs):
        """
        Runs the algorithms to find assignments for the segments of frames, and fills self.annotations with resulting assignments.
        :param frames: which frames to find assignments for   # todo: could be string, like 'all'
        """
        raise NotImplementedError
