import logging
import sys
import itertools
from .. import helpers as h
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from ..GlobalParameters import GlobalParameters

from scipy import ndimage as ndi
#from skimage.morphology import watershed#MB removed to prevent future warnings
from skimage.segmentation import watershed#MB added
from skimage.feature import peak_local_max


class Segmenter:
    def __init__(self, data, segmentation_parameters):
        """
        Class to segment the movie.
        :param data: instance of Dataset
        :param segmentation_parameters: instance of Parameters with kind "segmentation"
        """
        self.data = data
        self.parameters = segmentation_parameters
        self.logger = logging.getLogger('Segmenter')

    def test_segmentation_parameters(self, frame=0, user_params=False):
        """
        Performs segmentation of the first frame using self.parameters(possibly updated by user)
        then plots resulting segments and intermediate steps.
        Segmentation results are NOT saved.
        :param frame: which time frame on which to test segmentation
        :param user_params: whether to ask the user to input parameter values before segmentation
            (if so, updates the parameters)   # Todo: useless in Core version
        """
        self.logger.debug(sys._getframe().f_code.co_name)
        if user_params:
            self.parameters.user_edit()
        self.logger.info("Segmenting first frame with parameters {}".format(self.parameters))
        cache = NeuronSegmentationCache()
        img = self.data.get_frame(frame)#MB: I think this returns the red channel frame
        neuron_segmentation2(img, cache=cache, **self.parameters, dimensions=GlobalParameters.dimensions)
        cache.plot_shed_overlay()
        cache.plot_pre_shed()

    def segment(self, frames_to_segment):
        """
        Performs segmentation of given frames (using self.parameters), and stores results in self.annotations.
        If some of the frames are already segmented, segmentation will be overwritten.   # Todo: remove corresponding features etc
        :param frames_to_segment: iterable of times to be segmented
        """
        # Todo: if loading images is very long, could be combined with feature extraction for instance by taking an argument images
        frames_to_segment = sorted(frames_to_segment)  # sorting is for fun
        self.logger.debug("Segmenting frames {} with parameters {}".format(frames_to_segment, self.parameters))
        for frames in h.batch(frames_to_segment):
            images = [self.data.get_frame(t, force_original=True) for t in frames]
            params = {"dimensions": GlobalParameters.dimensions, **self.parameters}
            segments = h.parallel_process(images, neuron_segmentation2, params)
            for t, segmented in zip(frames, segments):
                self.data.save_mask(t, segmented, force_original=True, update_nb_neurons=True)   # Todo: this will interfere with any pre-existing neurons
                # save_mask; creates  datasets of seg amd mask for the frame ans saves the segmentation in both of them
        # todo: if loading video is long, could also be parallelized?


class NeuronSegmentationCache:
    """
    Cache object which stores the intermediate steps of the neuron_segmentation process
    contains all the individual steps of a segmentation as properties

    methods
    * plot_pre_shed makes plots of the blurring and thresholding and distance transform
    * plot_shed_overlay displays the final shedded overlay with the original image
    """

    def __init__(self):
        self.im = None   # original frame   # 1
        self.thresholded = None   # thresholded perc most brightest after smoothing   # 3
        self.markers = None   # seeds for watershed
        self.prestick_shed = None
        self.poststick_shed = None
        self.postminvol_shed = None   # final result (only region numbering may differ)
        self.dtr = None   # after distance transform   # 4
        self.sm_dtr = None   # smoothed after distance transform   # 5
        self.sm = None   # smoothed for background removal   # 2

    def plot_pre_shed(self):
        """ Plot the data before any watershed has been applied (sm, dtr) """
        h.quick_project_imshow(self.im, 'Original image', False)
        h.quick_project_imshow(self.sm, 'Blurred and sharpened image', False)
        h.quick_project_imshow(self.sm_dtr, 'Smoothed distance transform', False)
        h.quick_project_imshow(self.dtr, 'Unsmoothed distance transform', False)

        # Plot the overlay of threshold and seed markers
        plt.figure(figsize=(8, 6))
        plt.title("Thresholded")
        plt.imshow(h.project(self.thresholded, 2).T, aspect='auto', origin="lower")

        # Find the locations of the seeds
        marker_proj = h.project(self.markers, 2)
        locs = np.empty((2, np.count_nonzero(self.markers)))
        count = 0

        for x in range(marker_proj.shape[0]):
            for y in range(marker_proj.shape[1]):
                if marker_proj[x, y] > 0:
                    locs[0, count] = y
                    locs[1, count] = x
                    count += 1
        plt.scatter(locs[1], locs[0], color='red')
        plt.show()

    def plot_shed_overlay(self, shed=None):
        """ overlay the chosen shed property with the original image stored in self.im """
        if shed is None:
            shed = self.postminvol_shed
        h.quick_project_imshow(shed > 0, 'Segmented regions with watershed')
        # plot_segs_overlay(shed, self.im, title_prefix=title_prefix)
        # todo: restore last line, or plot with different colors?

#        cache.prestick_shed = shed
#        cache.poststick_shed = shed
#        cache.postminvol_shed = shed


########################################################################################################################
# The main segmentation function
# Todo: if class methods can be pickled, then put these into the class

def neuron_segmentation2(im, sigm=1, bg_factor=25, perc=.99, sigm_dtr=5, min_dist=1, st_factor=.2, minvol=1,
                         min_pixels_object=120, large_obj_threshold=500, max_pixels_object=100000, dist_threshold=10,
                         dimensions=(.1625, .1625, 1.5), cache=None):
    """
    Improved segmentation of neurons

    im:         Image to be segmented
    sigm:       Sigma for initial smooting of image
    bg_factor:  Factor for smoothing in background removal
    perc:       Percentile for thresholding
    sigm_dtr:   Sigma for smoothing of distance transform
    min_dist:   Minimal distance in x and y direction between two maxima in
                seed detection in µm.
    st_factor:  Factor for sticking together volumes
    minvol:     Minimum cell volume in µm^3
    min_pixels_object: Minimum size of an object in terms of pixels after thresholding
    large_obj_threshold: Threshold for objects to be considered a "large" object when removing outliers that are far
                         away from everything else
    dist_threshold: Minimum distance of a small object to the nearest large object for it to be considered an outlier
    dimensions: (xysize, xysize, zsize) with xysize the pixel size along x and y direction and zsize the pixel size
                    along z direction (both in µm)
    cache:      cache object of type NeuronSegmentationCache which stores the intermediate steps of this process for later analysis


    Algorithm:
    1.  Smooths image with Gaussian filter with parameter sigm
    2.  Smooths image with Gaussian filter with parameter sigm*bg_factor
    3.  Subtracts second image from first to yield a smooth image with
        sharpened borders.

        NOTE: Observation shows that this works, but no quantitative assessement has been done
    4.  Thresholds resulting image according to the percentile determined by
        perc
    5.  Calculates distance transform
    6.  Smooths distance transform with Gaussian filter with parameter sigm_dtr
        to remove small irregularities.
    7.  Finds local maxima with constraint that local maxima have to be spread
        out with a minimal distance determined by min_dist
    8.  Uses found local maxima as seeds for watershed algorithm, which
        segments the image based on the smoothed watershed transform
    9.  For every segment which are immediate neighbors, determine the number
        of neighboring pixels and the volume of the smaller of the two
        segments. If n_neighbors / volume**(2/3) > st_factor, stick the two
        segments together. This is based on the heuristic that if two segments
        are highly connected, they actually are one segment that has mistakenly
        been divided into two.

        NOTE: Effectiness of this step has not been tested
    10. Remove all segments which have a volume less than minvol (in µm).
    """
    xysize, xysize2, zsize = dimensions
    assert xysize == xysize2, "Pixels should have same dimension along x and y"
    if not cache is None:
        cache.im = im

    # Smooth image, remove glow background
    sdev = np.array([sigm, sigm, sigm * xysize / zsize])
    sm = ndi.gaussian_filter(im, sigma=sdev) - \
         ndi.gaussian_filter(im, sigma=sdev * bg_factor)

    if not cache is None:
        cache.sm = sm

    # Thresh
    thr = sm > np.quantile(sm, perc)
    if not cache is None:
        cache.thresholded = thr

    # Remove small objects
    thr = get_components_image(thr, min_pixels_object=min_pixels_object, max_pixels_object=max_pixels_object).astype(int)
    # Remove small objects that are far from any large object
    thr, _ = remove_objects_noise(thr, large_obj_threshold=large_obj_threshold, dist_threshold=dist_threshold)

    # Dist transform, seeds
    dtr = ndi.morphology.distance_transform_edt(thr, sampling=[xysize, xysize, zsize])
    if not cache is None:
        cache.dtr = dtr
    sm_dtr = ndi.gaussian_filter(dtr, sigma=[sigm_dtr, sigm_dtr, sigm_dtr * xysize / zsize])
    if not cache is None:
        cache.sm_dtr = sm_dtr

    # Maxima based seed
    min_dist_px = int(min_dist // xysize)
    max_footprint = np.ones((2 * min_dist_px + 1, 2 * min_dist_px + 1, 3))
    #loc_max = peak_local_max(sm_dtr, footprint=max_footprint, exclude_border=False, indices=False)#MB commented this out and added the following lines to prevent future warning

    loc_max = peak_local_max(sm_dtr, footprint=max_footprint, exclude_border=False)#MB added
    Mid_img = np.zeros(np.shape(sm_dtr))#MB added
    for i in range(len(loc_max[:,0])):#MB added
        Mid_img[loc_max[i,0],loc_max[i,1],loc_max[i,2]] = 1#MB added
    loc_max = Mid_img#MB added

    markers = ndi.label(loc_max)[0]
    if not cache is None:
        cache.markers = markers

    # Watershed
    shed = np.array(watershed(-sm_dtr, markers, mask=thr),dtype=np.uint8)
    if not cache is None:
        cache.prestick_shed = np.copy(shed) # SJR: I added np.copy since prestick_shed was being updated with each change in "shed"

    # Stick together
    stick_together(shed, st_factor)
    if not cache is None:
        cache.poststick_shed = np.copy(shed)

    # Filter by size
    for m in np.unique(shed):
        if (shed == m).sum() * xysize * xysize * zsize < minvol:
            shed[shed == m] = 0

    if not cache is None:
        cache.postminvol_shed = np.copy(shed)

    # Reorder numbers to not have gaps
    out = np.zeros(shed.shape, dtype=np.uint8)
    uniqm = np.unique(shed)
    for i, m in zip(range(len(uniqm)), uniqm):
        out[shed == m] = i

    return out


def get_components_image(im, min_pixels_object=120, max_pixels_object=100000):
    """
    Get the connected components in an image, where the connected components
    are determined by the nonzero pixels. Remove (i.e., zero) all components
    that have fewer than min_pixels_object pixels.
    TODO: Might want to use the physical size rather than the number of
    pixels later
    """
    # Get the connected components
    labeled, _ = ndi.label(im)
    # Get the size of each connected component
    all_obj_sizes = np.bincount(labeled.ravel())
    # Determine which components are larger than the threshold
    good_obj_nums = np.where(all_obj_sizes > min_pixels_object)[0]

    # Create a new array where pixels are zeroed if they don't belong to objects
    # above the given size. The new labels will go from 1-# objects
    im_new = np.zeros((labeled.shape))
    for new_obj_num, obj_num in enumerate(good_obj_nums):
        if all_obj_sizes[obj_num] < max_pixels_object:
            good_idxs = np.where(labeled == obj_num)
            im_new[good_idxs] = new_obj_num+1

    return im_new


def remove_objects_noise(img, large_obj_threshold=500, dist_threshold=10, dimensions=(.1625, .1625, 1.5)):
    """
    Remove objects that are both smaller than large_obj_threshold and are further than dist_threshold from the nearest
    large object.
    """
    obj_sizes = np.bincount(img.flatten())
    largest_idxs = np.where(obj_sizes > large_obj_threshold)[0][1:]  # Omit 0
    if len(largest_idxs) == 0:
         print('Warning: No objects of size larger than ' + str(large_obj_threshold) + ' and less than the given'
                         ' max_pixels_object found')
         return img, []
    smallest_idxs = np.where(obj_sizes < large_obj_threshold)[0]
    largest_idxs_locs = np.array(np.where(img * np.isin(img, largest_idxs))).T * dimensions
    all_min_dists = []
    for idx in smallest_idxs:
        if obj_sizes[idx] > 0:
            object_locs = np.array(np.where(img * (img == idx))).T * dimensions
            dists = scipy.spatial.distance.cdist(object_locs, largest_idxs_locs)
            min_dist = np.min(dists)
            all_min_dists.append(min_dist)
            if min_dist > dist_threshold:
                img = img * (img != idx)

    return img, all_min_dists


def stick_together(seg, factor, connectivity=1):
    """
    For every segment which are immediate neighbors, determine the number
    of neighboring pixels and the volume of the smaller of the two
    segments. If n_neighbors / volume**(2/3) > factor, stick the two
    segments together. This is based on the heuristic that if two segments
    are highly connected, they actually are one segment that has mistakenly
    been divided into two.

    THIS IS DONE IN PLACE.

    seg:            Segmented image
    factor:         Cutoff factor for sticking two parts together
    connectivity:   Connectivity for neighbor count. Default is set to consider
                    a 1-connected neighborhood in xy, and no neighbors in z.
    """
    pot_neighbors = get_potential_neighbors(seg)

    for s1, s2 in pot_neighbors:
        nb_neighbors = get_nb_neighbors(seg == s1, seg == s2,
                                        connectivity=connectivity)
        if nb_neighbors == 0:
            # No neighbors
            continue

        vol_s1 = (seg == s1).sum()
        vol_s2 = (seg == s2).sum()

        if factor < nb_neighbors / min(vol_s1, vol_s2) ** (2 / 3):
            seg[seg == s2] = s1

    return seg


def get_potential_neighbors(seg):
    """
    From a segmented region, identifies which of the segments are potentially
    next to each other.
    :param seg:     Segmentation, nd-array with integer labels corresponding
                    to segment
    :return:    List of potential neighbors as tuples
    """
    conncomp = ndi.label(seg)[0]
    pot_neighbors = []
    for lab in np.unique(conncomp):
        groups = np.unique(seg[conncomp==lab])
        pot_neighbors += list(itertools.combinations(groups, 2))
    return pot_neighbors


def get_nb_neighbors(bin1, bin2, neighborhood=np.ones((3, 3, 3)), connectivity=None):
    """
    Counts numbers of neighbors of two binary images. Make sure that none of
    the labeled pixels of the binary images overlap for sensible results.
    :param bin1:    Binary image
    :param bin2:    Binary image
    :param neighborhood:    Defines neighborhood for consideration.
    :param connectivity:    Can be used to create binary structure of given
                            connectivity. Overrides neighborhood.
    :return:        Number of pixels in binary image 2 that border on
                    binary image 1.
    """
    if connectivity is not None:
        neighborhood = ndi.morphology.generate_binary_structure(2, connectivity)
        neighborhood = np.atleast_3d(neighborhood)

    nx, ny, nz = neighborhood.shape

    shifts = np.argwhere(neighborhood)
    shifts[:, 0] -= nx // 2
    shifts[:, 1] -= ny // 2
    shifts[:, 2] -= nz // 2

    nb_neighbors = 0
    for vector in shifts:
        shifted = binary_translate_3d(bin2, vector)
        nb_neighbors += np.logical_and(bin1, shifted).sum()

    return nb_neighbors


def binary_translate_3d(image, vector):
    """
    Shifts binary 3d image according to vector of integers. Undefined
    pixels are set to zero.
    :param image:   Image to shift in 3d space
    :param vector:  Vector of integers along which to move the image.
    :return:        Shifted image
    """
    vector = vector.astype(int)
    x,y,z = vector
    nx, ny, nz = image.shape

    xshift = np.zeros(image.shape, dtype=bool)
    if x==0:
        xshift = image
    elif x>0:
        xshift[x:,:,:] = image[:-x,:,:]
    elif x<0:
        xshift[:x,:,:] = image[-x:,:,:]

    yshift = np.zeros(image.shape, dtype=bool)
    if y==0:
        yshift = xshift
    elif y>0:
        yshift[:,y:,:] = xshift[:,:-y,:]
    elif y<0:
        yshift[:,:y,:] = xshift[:,-y:,:]

    out = np.zeros(image.shape, dtype=bool)
    if z==0:
        out = yshift
    elif z>0:
        out[:,:,z:] = yshift[:,:,:-z]
    elif z<0:
        out[:,:,:z] = yshift[:,:,-z:]

    return out
