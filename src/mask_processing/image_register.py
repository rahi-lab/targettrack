import warnings
import numpy as np
from tqdm import tqdm
import gmmreg._core as core
#print("HOTFIX NEEDED: this is not pip installable, we should make a setup.py if we need non-pip install packages or either do a relative import")
import matplotlib.pyplot as plt
import copy
from scipy.optimize import fmin_l_bfgs_b
from mpl_toolkits.mplot3d import Axes3D

from skimage.measure import find_contours
from ..helpers import helpers as h
from itertools import repeat
from ..datasets_code.EPFL_datasets.nd2Data import nd2Data

# stop matplotlib debug-level spam
plt.set_loglevel("info")
ParallelCompute=True#whether to compute the transformation parallely or not

class Register_Rotate:

    # Todo: why nd2Data not DataSet?
    def __init__(self, data: nd2Data, ref_frames=None, frames_to_register=None, num_samples_per_neuron=10):
        """
        Assign (in data) the best possible translation and Rotation in 2D to align frames_to_rotate on reference frames.
        :ref_frames: which frames to use as references; if not provided, will get previous reference frames from data;
            if there are none, will use first frame.
        :param frames_to_register: which frames to register; all if not provided
        :param num_samples_per_neuron: Integer, number of sample contour points needed for registration.
        """
        self.num_samples_per_neuron = num_samples_per_neuron
        self.data = data
        if ref_frames is None:
            ref_frames = self.data.ref_frames()
        if not len(ref_frames):
            warnings.warn("No reference provided and none already used; using first frame as reference.")
            ref_frames = [0]
        self.ref_frames = list(ref_frames)
        if frames_to_register is None:
            frames_to_register = self.data.frames
        all_transformed = self.data.get_transformation_keys()#the list of frames that have transformation matrix saved
        self.ref_frames_remaining = list(set(self.ref_frames) - set(all_transformed))
        self.frames_to_register = list(set(frames_to_register)- set(self.ref_frames))# MB: changed  and removed - set(all_transformed)  to register frames multiple times
        self.segments = dict()
        orig_ref = self.data.base_ref_frame()#TODO MB: solve the issue with orig ref assignment
        if orig_ref is None:
            orig_ref = self.ref_frames[0]
        if orig_ref not in self.ref_frames:
            warnings.warn("Cannot use same base reference frame as has been used before, which is {}.".format(orig_ref)
                          + " This can cause some inconsistencies in the alignment of images.")
            orig_ref = self.ref_frames[0]#MB changed orig_ref from 0 to -1
        self.original_reference = orig_ref

        self.assign_identity(self.original_reference,is_ref=True)
        self.load_all_references()
        self.prerotate_reference()
        self.transform_batch()

    def prerotate_reference(self):
        """
        Compute the transformations of the references wrt to the original reference.
        """
        for frame in self.ref_frames_remaining:
            self.transform_one(frame, [self.original_reference])
        return

    def transform_batch(self):
        """
        Compute the transformations in batches ( a tradeoff wrt storing segment all segements in memory )
        """
        batches = h.batch(self.frames_to_register)
        batches = list(batches)

        if ParallelCompute:
            #MB: ref_segs and origRef are computed outside of the function for parallelization
            ref_segs = [np.asarray(self.segments[r]) for r in self.ref_frames]
            origRef = self.original_reference
            trans_to_refs = {refFr: self.data.get_transformation(refFr) for refFr in self.ref_frames}
            for batch in batches:
                non_ref = set(batch) - set(self.ref_frames)
                all_frames = list(set(non_ref) - set(self.segments.keys()))#MB: is this line necessary?
                images = [int(time) for time in non_ref]
                segs = [self.load_single_contour(time) for time in non_ref]
                #MB: make a sequence of all the inpputs of the function batch:
                sequence = zip(images,repeat(self.ref_frames),repeat(ref_segs),repeat(origRef),repeat(trans_to_refs),segs)
                result = h.parallel_process2(sequence, transform_one)
                for t, res in zip(non_ref, result):
                    self.data.save_ref(t, res[1])
                    self.data.save_score(t, res[2])
                    self.data.save_transformation_matrix(t, res[0])
        else:
            for batch in batches:
                non_ref = set(batch) - set(self.ref_frames)
                all_frames = list(set(non_ref) - set(self.segments.keys()))
                segments = {t: self.load_single_contour(t) for t in all_frames}
                for frame in tqdm(non_ref):
                    self.transform_one(frame, self.ref_frames, segments=segments[frame])
                del segments


    def load_all_references(self):
        """
        Loads countours for the reference and stores them in memory( used repeatedly hence stored in memory ).
        """
        self.segments = {t: self.load_single_contour(t) for t in self.ref_frames}

    def transform_one(self, frame, ref_frames, segments=None):
        """
        Computes and stores the translation and rotation of the frames w.r.t reference frames
        :param frame: Integer, the time frame to compute transformation
        :param ref_frames: The list of reference frames to compute.
        :param segments: the sample of the contour to register
        """
        if segments is None:
            points = self.load_single_contour(frame)
        else:
            points = segments
        if len(points) == 0:
            self.assign_identity(frame)
            return
        points = np.asarray(points)
        ctrl_pts = points
        min_frame_value = 100
        min_frame = frame

        transform = None
        for ref_frame in ref_frames:
            ref_seg = np.asarray(self.segments[ref_frame])
            points, ref_seg, after_tps, func_min = self.registration_JV2D(points, ref_seg, ctrl_pts)
            Initial = points
            final = after_tps
            transform_temp, loss_affine = self.rotation_translation(final, Initial)
            if (10 * func_min + loss_affine < min_frame_value):
                min_frame_value = 10 * func_min + loss_affine
                transform = copy.deepcopy(transform_temp)
                min_frame = ref_frame

        self.data.save_ref(frame, min_frame)#save the reference frames coresponding to the current frame
        self.data.save_score(frame, min_frame_value)

        if min_frame != self.original_reference:
            ref_to_orig = self.data.get_transformation(min_frame)
            transform = self.composite_transform(ref_to_orig, transform)

        self.data.save_transformation_matrix(frame, transform)

    def composite_transform(self, ref_to_orig, image_to_ref):
        """
        Compose two both translations and rotations and return a composed transformation
        :param ref_to_orig: (3,4) nparray, translation from original reference to the best reference
        :param image_to_ref: (3,4) nparray, translation from current frame to best reference selected.
        :return:
        """
        rot_ref_to_orig = ref_to_orig[:, :3]
        rot = image_to_ref[:, :3]
        offset = image_to_ref[:, 3]
        offset_ref_to_orig = ref_to_orig[:, 3]
        rot_final = rot @ rot_ref_to_orig
        offset_final = rot @ offset_ref_to_orig + offset
        transform = np.zeros((3, 4))
        transform[:, :3] = rot_final
        transform[:, 3] = offset_final
        return transform

    def assign_identity(self, frame, is_ref = False):
        """
        Assigns a identity for the transformation in case it has no neurons.
        """
        transform_matrix = np.identity(3)
        offset = np.zeros((3, 1))
        transform_matrix = np.append(transform_matrix, offset, axis=1)
        self.data.save_transformation_matrix(frame, transform_matrix)
        if is_ref:
            self.data.save_score(frame, 0)
            self.data.save_ref(frame, frame)
        else:
            self.data.save_score(frame, -1)
            self.data.save_ref(frame, -1)

    def display3Dpointsets(self, A, B, ax):
        """
        Scatter plots to display 3D point sets
        :param A: nparray of dimesnsion (n,3)
        :param B: nparray of dimesnsion (n,3)
        :param ax: Axis3D, position in the 3D plot
        """
        ax.scatter(A[:, 0], A[:, 1], A[:, 2], c='y', marker='o')
        ax.scatter(B[:, 0], B[:, 1], B[:, 2], c='b', marker='+')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    def display2Dpointsets(self, A, B, ax):
        """
        Scatter plots to display 2D point sets
        :param A: nparray of dimesnsion (n,3)
        :param B: nparray of dimesnsion (n,3)
        :param ax: Axis3D, position in the 3D plot
        """
        ax.scatter(A[:, 0], A[:, 1], c='y', marker='o')
        ax.scatter(B[:, 0], B[:, 1], c='b', marker='+')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')


    def displayABC(self, A, B, C, fname):
        """
        Display two images to compare the first two A,B and next two B,C.
        :param A: nparray of dimesnsion (n,3)
        :param B: nparray of dimesnsion (n,3)
        :param C: nparray of dimesnsion (n,3)
        :param fname: string, file name to store the plot , will show if it is None.
        """
        fig = plt.figure()
        dim = A.shape[1]
        if dim == 3:
            plot1 = plt.subplot(1, 2, 1)
            ax = Axes3D(fig, rect=plot1.get_position())
            self.display3Dpointsets(A, B, ax)
            plot2 = plt.subplot(1, 2, 2)
            ax = Axes3D(fig, rect=plot2.get_position())
            self.display3Dpointsets(C, B, ax)
        if fname is not None:
            plt.savefig(fname)
        else:
            plt.show()

    def displayABC2D(self, A, B, C, fname):
        """
        Display two images to compare the first two A,B and next two B,C.
        :param A: nparray of dimesnsion (n,2)
        :param B: nparray of dimesnsion (n,2)
        :param C: nparray of dimesnsion (n,2)
        :param fname: string, file name to store the plot , will show if it is None.
        """
        fig = plt.figure()
        dim = A.shape[1]
        if dim == 2:
            ax = plt.subplot(1, 2, 1)
            #ax = plot1.get_position()
            self.display2Dpointsets(A, B, ax)
            ax = plt.subplot(1, 2, 2)
            #ax = plot2.get_position()
            self.display2Dpointsets(C, B, ax)
        if fname is not None:
            plt.savefig(fname)
        else:
            plt.show()

    def registration_JV(self, points, reference, ctrl_pts):
        """
        :param points: (n,3) numpy array, the model to register
        :param reference: (n,3) numpy array, the reference model
        :param ctrl_pts: (n,3) numpy array, sample points of the model used for registration.
        :return: points, reference and the transformed points and the registration loss.
        """
        level = 3
        scales = [ .3, .2, .01]# the scale parameters of Gaussian mixtures, from coarse to fine,
        lambdas = [ .01, .001, .001]# weights of the regularization term, e.g. the TPS bending energy
        iters = [500, 500, 500]# the max number of function evaluations at each level
        [points, c_m, s_m] = core.normalize(points)#translate and scale a point set so it has zero mean and unit variance
        [reference, c_s, s_s] = core.normalize(reference)
        [ctrl_pts, c_c, s_c] = core.normalize(ctrl_pts)
        after_tps, x0, loss = self.run_multi_level(points, reference, ctrl_pts, level, scales, lambdas, iters)
        points = core.denormalize(points, c_m, s_m)
        reference = core.denormalize(reference, c_s, s_s)
        after_tps = core.denormalize(after_tps, c_s, s_s)
        #self.displayABC(points, reference, after_tps,str(loss)+".png")
        return points, reference, after_tps, loss

    def registration_JV2D(self, points, reference, ctrl_pts):
        """
        :param points: (n,3) numpy array, the model to register
        :param reference: (n,3) numpy array, the reference model
        :param ctrl_pts: (n,3) numpy array, sample points of the model used for registration.
        :return: points, reference and the transformed points and the registration loss.
        """
        points = points[:,:2]
        reference = reference[:,:2]
        ctrl_pts = ctrl_pts[:,:2]

        level = 4#the levels changed from 3 to 4
        scales = [.6, .3, .2, .1]# the scale parameters of Gaussian mixtures, from coarse to fine,
        lambdas = [0.1, .01, .001, .001]# weights of the regularization term, e.g. the TPS bending energy
        iters = [100, 100, 500, 300]# the max number of function evaluations at each level, changed itteer from 500 to 100
        [points, c_m, s_m] = core.normalize(points)#translate and scale a point set so it has zero mean and unit variance
        [reference, c_s, s_s] = core.normalize(reference)
        [ctrl_pts, c_c, s_c] = core.normalize(ctrl_pts)
        after_tps, x0, loss = self.run_multi_level(points, reference, ctrl_pts, level, scales, lambdas, iters)
        points = core.denormalize(points, c_m, s_m)
        reference = core.denormalize(reference, c_s, s_s)
        after_tps = core.denormalize(after_tps, c_s, s_s)
        #self.displayABC2D(points, reference, after_tps,str(loss)+".png")
        return points, reference, after_tps, loss

    def run_multi_level(self, model, scene, ctrl_pts, level, scales, lambdas, iters):
        """
        The point set registration by Jian Vemuri, check https://pubmed.ncbi.nlm.nih.gov/21173443/
        :param model:(n,3) numpy array, the reference model
        :param scene: (n,3) numpy array, the scene
        :param ctrl_pts: (n,3) control points to register
        :param level: Integer,
        :param scales: list of scales of length level, Gaussian variance at each level,
        :param lambdas: list of double of length level, Bending regularizer at each level.related to the energy of the nonlinear transformation
        :param iters: list of Integers of length level,Number of iterations to run for each level
        :return: the transformed points, also the registration loss
        """
        [n, d] = ctrl_pts.shape
        x0 = core.init_param(n, d)#initial parameters of thin plate spline
        [basis, kernel] = core.prepare_TPS_basis(model, ctrl_pts)#basis for performing TPS transformation
        loss = 1
        for i in range(level):
            x = fmin_l_bfgs_b(core.obj_L2_TPS, x0, None, args=(basis, kernel, scene, scales[i], lambdas[i]),
                              maxfun=iters[i])
            x0 = x[0]
            loss = x[1]
        after_tps = core.transform_points(x0, basis)
        return after_tps, x0, loss

    def rotation_translation(self, Initial, final):
        """
        compute the max x,y rotation R and translation T, such that final - R@Initial + offset
        :param Initial: (n,3) numpy array, where n is number of 3D points
        :param final: (n,3) numpy array, where n is number of 3D points.
        :return: the concatenated numpy array
        """
        Initial = Initial[:, :2]
        final = final[:, :2]
        c_i = np.mean(Initial, axis=0)
        Initial = Initial - c_i
        c_f = np.mean(final, axis=0)
        final = final - c_f
        H = Initial.T @ final
        U, D, V = np.linalg.svd(H)
        R = V.T @ U.T
        det = np.linalg.det(R)
        D = np.diag([1, 1, det])
        R = V.T @ (U.T)
        offset = c_f - np.dot(R, c_i)
        transform_points = np.matmul(Initial, R.T)
        loss = np.linalg.norm(final - transform_points) / np.linalg.norm(final)
        # offset = offset.reshape((offset.shape[0],1))
        d_3_transform = np.zeros((3, 4))
        d_3_transform[:2, :2] = R
        d_3_transform[2, 2] = 1.0
        d_3_transform[:2, 3] = offset
        return d_3_transform, loss


    def add_good_to_ground_truth(self, n_display=6):
        all_dists = {t: self.data.get_score(t) for t in self.frames_to_register}
        # first, histogram:
        plt.figure()
        distances = all_dists.values()
        binsize = 3
        plt.hist(distances, bins=np.arange(min(distances), max(distances), binsize))
        plt.xlabel("Registration distance")
        plt.ylabel("Nb of frames")
        plt.title("Histogram of min registration distances for all frames.")
        # then, registration plots for n_display the frames:
        times, dists = [list(tpl) for tpl in zip(*sorted(all_dists.items(), key=lambda x: x[1]))]
        step = len(times) // max(n_display - 1, 1)
        plot_times = times[::step]
        if plot_times[-1] != times[-1]:
            plot_times.append(times[-1])
        # Todo: better UI than plt plots!!
        for t in plot_times:
            fig, axs = plt.subplots(1, 2)
            fig.suptitle("Time {} with registation loss {:.3g}".format(t, all_dists[t]))
            ref_frame = self.data.get_ref_frame(t)
            axs[0].imshow(self.data.get_frame(ref_frame).sum(axis=2))
            axs[1].imshow(self.data.get_frame(t).sum(axis=2))
        plt.show()

        # choose threshold according to plots
        threshold = input(  # Todo: better UI than input!!
            "Choose threshold distance (frames with registration distance below threshold will be added to ground truth)")
        threshold = float(threshold)
        # add the good frames from initial registration to ground truth
        good_times = set()
        for t, dist in all_dists.items():
            if dist <= threshold:
                good_times.add(t)
        self.data.flag_as_gt(good_times)


    # utils to compute the contours of segments

    @classmethod
    def find_3D_contour(cls, segment_binary):
        """
        Finds the 3d contour of a neuron.
        :param segment_binary: binary mask of one neuron
        :return: np array [[x1, y1, z1],...], a list of coordinates of points of the contour
        """
        contour_points = []
        for z in range(segment_binary.shape[2]):
            z_conts = find_contours(segment_binary[:, :, z], 0.5)
            for cont in z_conts:  # several contours
                for pt in cont:  # all points in contour
                    contour_points.append([*pt, z])
        return contour_points

    def sample_points_from_contour(self, contour):
        """
        Returns coordinates of points from the segment to be used for point set registration.
        :param contour: array [[x1, y1, z1],...], a list of the coordinates of all points of the contour.
        :return: list [[x1, y1, z1],...], a list of coordinates of approx n_points points from the contour
        """
        n_points = self.num_samples_per_neuron
        sampled_points = []
        contour = np.asarray(contour)
        all_xs = np.unique(contour[:, 0])  # is sorted
        x_step = int(np.ceil(len(all_xs) / np.sqrt(n_points / 2)))
        sample_xs = all_xs[(x_step - 1) // 2::x_step]
        for x in sample_xs:
            sample_x = contour[contour[:, 0] == x]
            all_ys = np.unique(sample_x[:, 1])
            y_step = int(np.ceil(len(all_ys) / np.sqrt(n_points / 2)))
            sample_ys = all_ys[(y_step - 1) // 2::y_step]
            for y in sample_ys:
                sample_y = sample_x[sample_x[:, 1] == y]
                mi = min(sample_y[:, 2])
                sampled_points.append([x, y, mi])
                ma = max(sample_y[:, 2])
                if ma != mi:
                    sampled_points.append([x, y, ma])
        return sampled_points

    def contour_of_segment(self, segdf):
        """
        :param segdf: numpy array, the mask of the segment
        :return:
        """
        countor = self.find_3D_contour(segdf)
        segdf = self.sample_points_from_contour(countor)
        return segdf

    def load_single_contour(self, frame):
        """
        Returns a sample of points from the contour of the segments,
        loads the entire segment file and filters the segment, to load for a set of frames use the load batch instead.
        :param frame: Integer, the time
        """
        frame_segment = self.data.segmented_frame(frame)
        if len(np.unique(frame_segment))<3:#MB added to avoid error when facing empty frames
            frame_segment = self.rotseg
        segments_in_frame = np.unique(frame_segment)
        points = []
        for seg in segments_in_frame:
            segdf = frame_segment == seg
            segdf = self.contour_of_segment(segdf.astype(int))
            if len(points) == 0:
                points = segdf
            else:
                points = np.append(points, segdf, axis=0)
        self.rotseg = frame_segment
        return points


########################################################################################################################
# The main alignment functions repeated here for parallelization
# Todo: if class methods can be pickled, then put these into the class

def transform_one(frame, ref_frames,ref_segs,origRef,trans_to_refs, segments=None):
    """
    Computes and stores the translation and rotation of the frames w.r.t reference frames
    :param frame: Integer, the time frame to compute transformation
    :param ref_frames: The list of reference frames to compute.
    :param segments: the sample of the contour to register
    """
    if segments is None:
        points = load_single_contour(frame)
    else:
        points = segments
    if len(points) == 0:
        transform,min_frame,min_frame_value =assign_identity(frame)

        return transform,min_frame, min_frame_value

    points = np.asarray(points)
    ctrl_pts = points
    min_frame_value = 100
    min_frame = frame

    transform = None
    for r in range(len(ref_frames)):
        ref_frame=ref_frames[r]
        ref_seg = ref_segs[r]

        points, ref_seg, after_tps, func_min = registration_JV2D(points, ref_seg, ctrl_pts)
        Initial = points
        final = after_tps
        transform_temp, loss_affine = rotation_translation(final, Initial)

        if (10 * func_min + loss_affine < min_frame_value):
            min_frame_value = 10 * func_min + loss_affine
            transform = copy.deepcopy(transform_temp)
            min_frame = ref_frame

    #MB TODO:Do we have to compute everything w.r.t. one single ref frame?
    if min_frame != origRef:#MB: is this necessary?
        #print("for frame"+str(frame)+"min_frame "+str(min_frame)+" doesn't match original ref "+str(origRef))
        ref_to_orig = trans_to_refs[min_frame]#self.data.get_transformation(min_frame)
        transform = composite_transform(ref_to_orig, transform)

    return transform,min_frame, min_frame_value

def composite_transform(ref_to_orig, image_to_ref):
    """
    Compose two both translations and rotations and return a composed transformation
    :param ref_to_orig: (3,4) nparray, translation from original reference to the best reference
    :param image_to_ref: (3,4) nparray, translation from current frame to best reference selected.
    :return:
    """
    rot_ref_to_orig = ref_to_orig[:, :3]
    rot = image_to_ref[:, :3]
    offset = image_to_ref[:, 3]
    offset_ref_to_orig = ref_to_orig[:, 3]
    rot_final = rot @ rot_ref_to_orig
    offset_final = rot @ offset_ref_to_orig + offset
    transform = np.zeros((3, 4))
    transform[:, :3] = rot_final
    transform[:, 3] = offset_final
    return transform

def assign_identity(frame, is_ref = False):
    """
    Assigns a identity for the transformation in case it has no neurons.
    """
    transform_matrix = np.identity(3)
    offset = np.zeros((3, 1))
    transform_matrix = np.append(transform_matrix, offset, axis=1)
    if is_ref:
        minfr_val=0
        minfr=frame
    else:
        minfr_val=-1
        minfr=-1

    return transform_matrix,minfr,minfr_val


def registration_JV(points, reference, ctrl_pts):
    """
    :param points: (n,3) numpy array, the model to register
    :param reference: (n,3) numpy array, the reference model
    :param ctrl_pts: (n,3) numpy array, sample points of the model used for registration.
    :return: points, reference and the transformed points and the registration loss.
    """
    level = 3
    scales = [ .3, .2, .01]# the scale parameters of Gaussian mixtures, from coarse to fine,
    lambdas = [ .01, .001, .001]# weights of the regularization term, e.g. the TPS bending energy
    iters = [500, 500, 500]# the max number of function evaluations at each level
    [points, c_m, s_m] = core.normalize(points)#translate and scale a point set so it has zero mean and unit variance
    [reference, c_s, s_s] = core.normalize(reference)
    [ctrl_pts, c_c, s_c] = core.normalize(ctrl_pts)
    after_tps, x0, loss = self.run_multi_level(points, reference, ctrl_pts, level, scales, lambdas, iters)
    points = core.denormalize(points, c_m, s_m)
    reference = core.denormalize(reference, c_s, s_s)
    after_tps = core.denormalize(after_tps, c_s, s_s)
    #self.displayABC(points, reference, after_tps,str(loss)+".png")
    return points, reference, after_tps, loss

def registration_JV2D(points, reference, ctrl_pts):
    """
    :param points: (n,3) numpy array, the model to register
    :param reference: (n,3) numpy array, the reference model
    :param ctrl_pts: (n,3) numpy array, sample points of the model used for registration.
    :return: points, reference and the transformed points and the registration loss.
    """
    points = points[:,:2]
    reference = reference[:,:2]
    ctrl_pts = ctrl_pts[:,:2]

    level = 4#the levels changed from 3 to 4
    scales = [.6, .3, .2, .1]# the scale parameters of Gaussian mixtures, from coarse to fine,
    lambdas = [0.1, .01, .001, .001]# weights of the regularization term, e.g. the TPS bending energy
    iters = [100, 100, 500, 300]# the max number of function evaluations at each level, changed itteer from 500 to 100
    [points, c_m, s_m] = core.normalize(points)#translate and scale a point set so it has zero mean and unit variance
    [reference, c_s, s_s] = core.normalize(reference)
    [ctrl_pts, c_c, s_c] = core.normalize(ctrl_pts)
    after_tps, x0, loss = run_multi_level(points, reference, ctrl_pts, level, scales, lambdas, iters)
    points = core.denormalize(points, c_m, s_m)
    reference = core.denormalize(reference, c_s, s_s)
    after_tps = core.denormalize(after_tps, c_s, s_s)
    #self.displayABC2D(points, reference, after_tps,str(loss)+".png")
    return points, reference, after_tps, loss

def run_multi_level(model, scene, ctrl_pts, level, scales, lambdas, iters):
    """
    The point set registration by Jian Vemuri, check https://pubmed.ncbi.nlm.nih.gov/21173443/
    :param model:(n,3) numpy array, the reference model
    :param scene: (n,3) numpy array, the scene
    :param ctrl_pts: (n,3) control points to register
    :param level: Integer,
    :param scales: list of scales of length level, Gaussian variance at each level,
    :param lambdas: list of double of length level, Bending regularizer at each level.related to the energy of the nonlinear transformation
    :param iters: list of Integers of length level,Number of iterations to run for each level
    :return: the transformed points, also the registration loss
    """
    [n, d] = ctrl_pts.shape
    x0 = core.init_param(n, d)#initial parameters of thin plate spline
    [basis, kernel] = core.prepare_TPS_basis(model, ctrl_pts)#basis for performing TPS transformation
    loss = 1
    for i in range(level):
        x = fmin_l_bfgs_b(core.obj_L2_TPS, x0, None, args=(basis, kernel, scene, scales[i], lambdas[i]),
                          maxfun=iters[i])
        x0 = x[0]
        loss = x[1]
    after_tps = core.transform_points(x0, basis)
    return after_tps, x0, loss

def rotation_translation(Initial, final):
    """
    compute the max x,y rotation R and translation T, such that final - R@Initial + offset
    :param Initial: (n,3) numpy array, where n is number of 3D points
    :param final: (n,3) numpy array, where n is number of 3D points.
    :return: the concatenated numpy array
    """
    Initial = Initial[:, :2]
    final = final[:, :2]
    c_i = np.mean(Initial, axis=0)
    Initial = Initial - c_i
    c_f = np.mean(final, axis=0)
    final = final - c_f
    H = Initial.T @ final
    U, D, V = np.linalg.svd(H)
    #R = V.T @ U.T
    #det = np.linalg.det(R)
    #D = np.diag([1, 1, det])
    R = V.T @ (U.T)
    offset = c_f - np.dot(R, c_i)
    transform_points = np.matmul(Initial, R.T)
    loss = np.linalg.norm(final - transform_points) / np.linalg.norm(final)
    # offset = offset.reshape((offset.shape[0],1))
    d_3_transform = np.zeros((3, 4))
    d_3_transform[:2, :2] = R
    d_3_transform[2, 2] = 1.0
    d_3_transform[:2, 3] = offset
    return d_3_transform, loss

def rotation_translation2(Initial, final):
    """
    compute the max x,y rotation R and translation T, such that final - R@Initial + offset
    this is more aligned with the results of the paper at:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=88573&tag=1
    :param Initial: (n,3) numpy array, where n is number of 3D points
    :param final: (n,3) numpy array, where n is number of 3D points.
    :return: the concatenated numpy array
    """
    Initial = Initial[:, :2]
    final = final[:, :2]
    c_i = np.mean(Initial, axis=0)
    Initial = Initial - c_i
    c_f = np.mean(final, axis=0)
    final = final - c_f
    H = final.T @ Initial
    U, D, V = np.linalg.svd(H)
    #R = V.T @ U.T
    R = U @ V
    det = np.linalg.det(R)
    S = np.diag([1, det])
    R = U @ S @ V.T
    offset = c_f - np.dot(R, c_i)
    transform_points = np.matmul(Initial, R.T)
    loss = np.linalg.norm(final - transform_points) / np.linalg.norm(final)
    # offset = offset.reshape((offset.shape[0],1))
    d_3_transform = np.zeros((3, 4))
    d_3_transform[:2, :2] = R
    d_3_transform[2, 2] = 1.0
    d_3_transform[:2, 3] = offset
    return d_3_transform, loss

def find_3D_contour(cls, segment_binary):
    """
    Finds the 3d contour of a neuron.
    :param segment_binary: binary mask of one neuron
    :return: np array [[x1, y1, z1],...], a list of coordinates of points of the contour
    """
    contour_points = []
    for z in range(segment_binary.shape[2]):
        z_conts = find_contours(segment_binary[:, :, z], 0.5)
        for cont in z_conts:  # several contours
            for pt in cont:  # all points in contour
                contour_points.append([*pt, z])
    return contour_points

def contour_of_segment(segdf):
    """
    :param segdf: numpy array, the mask of the segment
    :return:
    """
    countor = self.find_3D_contour(segdf)
    segdf = sample_points_from_contour(countor)
    return segdf

def load_single_contour(frame):
    """
    Returns a sample of points from the contour of the segments,
    loads the entire segment file and filters the segment, to load for a set of frames use the load batch instead.
    :param frame: Integer, the time
    """
    frame_segment = self.data.segmented_frame(frame)
    if len(np.unique(frame_segment))<3:#MB added to avoid error when facing empty frames
        frame_segment = self.rotseg
    segments_in_frame = np.unique(frame_segment)
    points = []
    for seg in segments_in_frame:
        segdf = frame_segment == seg
        segdf = contour_of_segment(segdf.astype(int))
        if len(points) == 0:
            points = segdf
        else:
            points = np.append(points, segdf, axis=0)
    self.rotseg = frame_segment
    return points
