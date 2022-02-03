import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import functools as ft
from ..parameters.GlobalParameters import GlobalParameters


def timed_func(process_name):
    """
    Adds printed time output for a function
    Will print out the time the function took as well as label this output with process_name

    :param process_name: human name of the process being timed
    """
    def decorator(f):
        def wrapper(*args, **kwargs):
            start = time.time()
            ret = f(*args, **kwargs)
            logging.getLogger("").info("Elapsed time for {}: {}".format(process_name, time.time()-start))
            return ret
        return wrapper
    return decorator


def batch(sequence, n=None):
    """
    split a sequence into batches of length n
    useful for chunking operations
    """
    chunksize = GlobalParameters.chunksize
    if n is None:
        n = chunksize
    l = len(sequence)
    for ndx in tqdm(range(0, l, n)):
        yield sequence[ndx:min(ndx + n, l)]


def parallel_process(sequence, func, params):
    n_processes = GlobalParameters.n_processes
    with multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.map(ft.partial(func, **params), sequence)
    return results

def parallel_process2(sequence, func):
    """MB:parallelizing a function with multiple arguments as asequence"""
    n_processes = GlobalParameters.n_processes
    with multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.starmap(func, sequence)
    return results

def project(img, axis):
    """
    Projects image onto axis by summing it up, for quick visualization.
    :returns: projected image
    """
    return np.sum(img, axis=axis)

def quick_project_imshow(img, title, show=True):
    """
    Project the volume along the z direction and plot it in it's own figure
    """
    f  = plt.figure()
    ax = plt.imshow(project(img, 2).T, aspect='auto', origin='lower')
    plt.title(title)
    if show:
        plt.show()

    return f, ax
