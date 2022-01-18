import logging
import warnings
import sys
from tqdm import tqdm
import numpy as np
from ..GlobalParameters import GlobalParameters

from scipy.spatial.distance import pdist


class FeatureBuilder:
    """Class to extract the features from the movie and annotation data"""
    def __init__(self, data, image_data=None):
        """
        :param data: instance of Dataset
        :param image_data: instance of ImageData. If provided, will whole-image data will be computed.
        """
        self.data = data
        self.image_data = image_data
        self.logger = logging.getLogger('FeatBuilder')

    def extract_features(self, frames):
        """

        :param frames: which time frames to extract features for.
        """
        self.logger.debug(sys._getframe().f_code.co_name)
        dimensions = GlobalParameters.dimensions
        for t in tqdm(frames):   # Todo: parallelize? (probably needs batches to get rid of classes)
            if self.data.use_seg_for_feature:#MB added to get features from both segmentations or mask matrix
                segmented = self.data.segmented_frame(t)
            else:
                segmented = self.data.get_mask(t, force_original=True)   # Todo: ideally, should work with segs instead of neurons
            # Check that there exist segments in the frame
            all_segs_bin = segmented != 0
            if not all_segs_bin.sum():
                warnings.warn("Nothing was found by segmentation in frame {}".format(t))
                continue
            im_red = self.data.get_frame(t)
            if self.image_data is not None:
                # get whole-frame, all-segments information (axes and center)
                all_segs_data = get_all_segs_data(all_segs_bin, dimensions)#coordinates of mask pixels
                self.image_data.assign_all_segs_data(t, all_segs_data)
                rawimage_data = get_rawimage_data(im_red, dimensions)
                self.image_data.assign_rawimage_data(t, rawimage_data)

            else:
                all_segs_data = None
                rawimage_data = get_rawimage_data(im_red, dimensions)#MB changed it from None
            # Calculate features
            for s in np.unique(segmented)[1:]:   # exclude  the first unique element which should always be 0
                # TODO: this behaviour must be changed:
                # use  all_segs_data or rawimage_data depending on whether "noseg" is used in ph.calculate_features:
                if len(np.argwhere(segmented == s))>2:#MB added the if condition to avoid errors
                    ftr_dict = calculate_features(segmented == s, im_red, dimensions, all_segs_info=rawimage_data)
                    self.data.save_features(t, s, ftr_dict)


########################################################################################################################
# Feature extraction helpers
# Todo: if class methods can be pickled, then put these into the class

def get_all_segs_data(all_segs_bin, dimensions):
    xyz_all = np.argwhere(all_segs_bin)
    total_center_of_mass = np.sum(xyz_all, axis=0) / len(xyz_all)
    xyz_all_centered = (xyz_all - total_center_of_mass) * dimensions
    axes = compute_principal_axes(xyz_all_centered.transpose())
    info = {"center_of_mass":total_center_of_mass, "axes":axes}
    return info


def compute_principal_axes(xyz_centered, weights=None, twodim=True):
    """

    :param xyz_centered: [list_of_xs, lst_of_ys, list_of_zs]
    :param weights: weights of each pixel
    :param twodim: whether to compute two main axes in xy plane, or three axes in 3D image.
    :return: ax1, ax2, (ax3 if not twodim else None)
    """
    if twodim:
        xyz_centered = xyz_centered[:2]
    cov = np.cov(xyz_centered, aweights=weights)#covariance between the variables x,y,z. pixels are the observations
    evals, evecs = np.linalg.eig(cov)#MB: it seems to be for finding the main axis of the worm
    # sort eigenvalues in decreasing order
    sort_indices = np.argsort(evals)[::-1]
    ax1 = evecs[:, sort_indices[0]]
    ax2 = evecs[:, sort_indices[1]]
    if twodim:
        ax3 = None
    else:
        ax3 = evecs[:, sort_indices[2]]
    return ax1, ax2, ax3


def get_rawimage_data(img, dimensions):
    """
    Compute whole-image characteristics from the raw red-channel image (no need for segmentation).
    :param img: one video frame (in red channel)
    :param dimensions: the scaling dimensions
    :return: whole-image characteristics dict
    """
    mi = np.percentile(img, 80)   # TODO: could be tuned
    ma = img.max()
    img = np.where(img >= mi, (img - mi) / (ma - mi), 0)
    xyz_all = np.nonzero(img)   # ([x1, x2, x3, ...], [y1, y2, y3, ...], [z1, z2, z3, ...])
    weights = img[xyz_all]
    xyz_all = np.vstack(xyz_all)
    center = np.average(xyz_all, axis=1, weights=weights)
    xyz_all_centered = (xyz_all - center[:, None]) * dimensions[:,None]
    axes = compute_principal_axes(xyz_all_centered, weights=weights)
    info = {"center_of_mass_noseg":center, "axes_noseg":axes}
    return info


def calculate_features(binary, im_red, dimensions, all_segs_info=None):
    """
    Extracts features for morphology in binary image, based on its shape and
    the underlying image. Extracted features: See below.
    Improved over version 1 in that it is invariant over translation.
    :param binary:  Binary mask of segment
    :param im_red:      Image used for segmentation, from which to compute features
    :param dimensions: The physical dimensions of the image np.array([dx, dy, dz])

    :return:        Dictionary of extracted features (see below)
    """

    vol = binary.sum()

    # weighted moment of inertia tensor
    xyz = np.argwhere(binary)
    if len(xyz) < 2:   # Todo: how can this happen??
        warnings.warn("Empty segment found")
        return {lab: np.nan for lab in ["Volume", "Red Total Intensity", "Red Intensity Var.", "Red Max. Intensity",
                                         "Weighted Ixx", "Weighted Iyy", "Weighted Izz", "Weighted Ixy", "Weighted Ixz",
                                         "Weighted Iyz", "elongation"]}   # Todo: what will happen to nans?
    m = im_red[xyz[:, 0], xyz[:, 1], xyz[:, 2]]

    centre_of_mass, wixx, wiyy, wizz, wixy, wixz, wiyz = calculate_moments(xyz, dimensions, m)

    if all_segs_info is not None:
        # TODO: use noseg or not
        total_center_of_mass = all_segs_info["center_of_mass_noseg"]   # TODO: make sure axes are computed using dimensions
        ax1, ax2, ax3 = all_segs_info["axes_noseg"]

        # relative location of segment
        rel_loc = (centre_of_mass - total_center_of_mass) * dimensions
        x_loc, y_loc, z_loc = rel_loc

        # rotation invariant relative location of segment
        if ax3 is None:
            rot_mat = np.linalg.inv(np.vstack([ax1, ax2]))#trransformation matrix to the coordinates where ax1 and ax2 are the x and y axis
            rot_inv_xy = np.matmul(xyz[:, :2], rot_mat)
            rot_inv_xyz = xyz.copy()
            rot_inv_xyz[:, :2] = rot_inv_xy
        else:
            rot_mat = np.linalg.inv(np.vstack([ax1, ax2, ax3]))
            rot_inv_xyz = np.matmul(xyz, rot_mat)
        rot_inv_centre_of_mass, rot_inv_wixx, rot_inv_wiyy, rot_inv_wizz, rot_inv_wixy, rot_inv_wixz, rot_inv_wiyz = calculate_moments(rot_inv_xyz, np.array([1,1,1]), m)#MB: why is the dimension set to [1,1,1]
        rot_inv_x_loc, rot_inv_y_loc, rot_inv_z_loc = rot_inv_centre_of_mass

    # elongation
    all_dists = pdist(xyz)
    diam = max(all_dists)
    elongation = diam / vol

    # Image properties
    total_intens = im_red[binary].sum()
    # avg_intens = im_red[binary].sum()/vol
    max_intens = im_red[binary].max()
    var_intens = im_red[binary].var()

    feature_dict =  {"Volume": vol
                     # , "Red Avg. Intensity": avg_intens
                     , "Red Total Intensity": total_intens
                     , "Red Intensity Var.": var_intens
                     , "Red Max. Intensity": max_intens
                     }


    feature_dict.update({"Weighted Ixx": wixx, "Weighted Iyy": wiyy
                         , "Weighted Izz": wizz, "Weighted Ixy": wixy
                         , "Weighted Ixz": wixz, "Weighted Iyz": wiyz
    })

    feature_dict.update({"elongation": elongation})

    if all_segs_info is not None:
        feature_dict.update({"Relative x location": x_loc, "Relative y location": y_loc, "Relative z location": z_loc,
                             "Rot. Inv. x loc": rot_inv_x_loc, "Rot. Inv. y loc": rot_inv_y_loc, "Rot. Inv. z loc": rot_inv_z_loc})
        feature_dict.update({"Rot. Inv. Weighted Ixx": rot_inv_wixx, "Rot. Inv. Weighted Iyy": rot_inv_wiyy,
                             "Rot. Inv. Weighted Izz": rot_inv_wizz, "Rot. Inv. Weighted Ixy": rot_inv_wixy,
                             "Rot. Inv. Weighted Ixz": rot_inv_wixz, "Rot. Inv. Weighted Iyz": rot_inv_wiyz})

    return feature_dict


def calculate_moments(xyz, dimensions, m, norm=1e9):
    #MB: I think m is the intensity of the pixels
    # Recenter coordinates around the center of mass
    centre_of_mass = np.sum(xyz, axis=0) / len(xyz)
    xyz_dist = (xyz - centre_of_mass) * dimensions  # why *dimensions?? it makes the wide dimension count more..??

    xyz2 = xyz_dist ** 2

    wixx = ((xyz2[:, 1] + xyz2[:, 2]) * m).sum() / norm
    wiyy = ((xyz2[:, 0] + xyz2[:, 2]) * m).sum() / norm
    wizz = ((xyz2[:, 0] + xyz2[:, 1]) * m).sum() / norm
    wixy = (xyz_dist[:, 0] * xyz_dist[:, 1] * m).sum() / norm
    wixz = (xyz_dist[:, 0] * xyz_dist[:, 2] * m).sum() / norm
    wiyz = (xyz_dist[:, 1] * xyz_dist[:, 2] * m).sum() / norm

    return centre_of_mass, wixx, wiyy, wizz, wixy, wixz, wiyz
