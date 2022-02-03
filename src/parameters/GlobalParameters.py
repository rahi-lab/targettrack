import numpy as np


class GlobalParameters:
    """
    Class to make available to all clients the value of global parameters such as maximum number of frames to load into
    memory and number of parallel processes.
    Meant to behave a bit like a singleton class, with all instances sharing the same class members so the latter can be
    accessed from anywhere without need to pass an existing instance.
    """
    chunksize = None
    n_processes = None
    dimensions = None

    # TODO: read and write from some config file

    @classmethod
    def set_params(cls, chunksize: int = None, n_processes: int = None, dimensions: tuple = None):
        """
        Sets the value of any one or several parameters. If no value is given, sets all to default.
        If any parameters are given, their value will be set and the others will be left unchanged.
        """
        if chunksize is None and n_processes is None and dimensions is None:
            # default values
            chunksize = 200
            n_processes = 8
            dimensions = (0.1625, 0.1625, 1.5)
        if chunksize is not None:
            cls.chunksize = chunksize
        if n_processes is not None:
            cls.n_processes = n_processes
        if dimensions is not None:
            cls.dimensions = np.array(dimensions)
