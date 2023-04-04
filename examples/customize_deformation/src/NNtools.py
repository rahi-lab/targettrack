import sys
import os
sys.path.append(os.path.split(__file__)[0])


import numpy as np

import scipy.ndimage as sim
from scipy.optimize import fmin_l_bfgs_b
from skimage.measure import find_contours
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1


def get_pts_from_masksJV(maskFrom,maskTo):
    '''
    MB added: applies Jian-Vermuri registration between points on the border of
    maskFrom and points on the border of maskTo
    maskFrom: mask asarray
    maskTo: maskArray
    returns:
    ptto: list of points on the border of mask objects of maskTo
    after_tps: transformed result of the set "points"
    '''

    ptfrom = load_single_contour_Mask(maskFrom)# the frame in the training set
    ptto = load_single_contour_Mask(maskTo)# the new augmented frame
    ctrl_pts = ptto
    points, ref_seg, after_tps, func_min = registration_JV(ptfrom, ptto, ctrl_pts)

    return points, after_tps, ptto

def registration_JV(points, reference, ctrl_pts):
    """
    :param points: (n,3) numpy array, the model to register
    :param reference: (n,3) numpy array, the reference model
    :param ctrl_pts: (n,3) numpy array, sample points of the model used for registration.
    :return: points, reference and the transformed points and the registration loss.
    after_tps: are the points after transformation to fit the ref
    """
    import gmmreg._core as core
    level = 4
    scales = [ .6, .3, .2, .1]
    lambdas = [ 0.1, .01, .001, .001]
    iters = [ 100, 100, 500, 300]
    [points, c_m, s_m] = core.normalize(points)
    [reference, c_s, s_s] = core.normalize(reference)
    [ctrl_pts, c_c, s_c] = core.normalize(ctrl_pts)
    after_tps, x0, loss = run_multi_level(points, reference, ctrl_pts, level, scales, lambdas, iters)
    points = core.denormalize(points, c_m, s_m)
    reference = core.denormalize(reference, c_s, s_s)
    after_tps = core.denormalize(after_tps, c_s, s_s)
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
    import gmmreg._core as core
    [n, d] = ctrl_pts.shape
    x0 = core.init_param(n, d)
    [basis, kernel] = core.prepare_TPS_basis(model, ctrl_pts)
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
    det_sign = np.prod(D) > 0
    D = np.diag([1, det_sign])
    R = V.T @ D @ (U.T)
    offset = c_f - np.dot(R, c_i)
    transform_points = np.matmul(Initial, R.T)
    loss = np.linalg.norm(final - transform_points) / np.linalg.norm(final)
    # offset = offset.reshape((offset.shape[0],1))
    d_3_transform = np.zeros((3, 4))
    d_3_transform[:2, :2] = R
    d_3_transform[2, 2] = 1.0
    d_3_transform[:2, 3] = offset
    return d_3_transform, loss

def find_3D_contour(segment_binary):
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


def sample_points_from_contour(contour):
    """
    Returns coordinates of points from the segment to be used for point set registration.
    :param contour: array [[x1, y1, z1],...], a list of the coordinates of all points of the contour.
    :return: list [[x1, y1, z1],...], a list of coordinates of approx n_points points from the contour
    MB: returns evenly distributed points on the xy projection of the contour
    """
    num_samples_per_neuron = 10
    n_points = num_samples_per_neuron
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

def contour_of_segment(segdf):
    """
    :param segdf: numpy array, the mask of the segment
    :return:
    """
    countor = find_3D_contour(segdf)
    segdf = sample_points_from_contour(countor)
    return segdf

def load_single_contour_Mask(mask):
    """
    Returns a sample of points from the contour of the segments,
    loads the entire segment file and filters the segment, to load for a set of frames use the load batch instead.
    :param frame: Integer, the time
    """
    frame_segment = mask
    segments_in_frame = np.unique(frame_segment)
    points = []
    for seg in segments_in_frame:
        if True:
            segdf = (np.array(frame_segment) == seg)#MB added np.array
            segdf = contour_of_segment(segdf.astype(int))
            if len(points) == 0:
                points = segdf
            else:
                points = np.append(points, segdf, axis=0)
    return points

def align_zs(mask_trans_rigid, target_pred_mask_i):
    _, _, nonzero_z_trans_rigid = np.nonzero(mask_trans_rigid)
    _, _, nonzero_z_target = np.nonzero(target_pred_mask_i)
    med_z_trans_rigid = np.median(nonzero_z_trans_rigid)
    med_z_target = np.median(nonzero_z_target)
    z_target_minus_trans = med_z_target - med_z_trans_rigid
    z_start, z_end = np.min(nonzero_z_target), np.max(nonzero_z_target)

    return z_start, z_end, z_target_minus_trans

def deformation_volume(mask,fr,target_pred_mask_i,target_fr_i,scale):
    # Perform Jian-Vemuri point-cloud registration to estimate a non-rigid (thin-plate spline) mapping from
    # stardardized points in the contours of objects in the training mask to standardized points in the
    # contours of objects in the target mask.
    # pts_train: the training set points that we are transforming
    # pts_train_trans: training set points after transformation
    # pts_target: goal of the transformation which is the predicted mask
    pts_train, pts_train_trans, pts_target = get_pts_from_masksJV(mask, target_pred_mask_i)


    # Compute the 2d translation + rotation matrix that best approximates the non-rigid transformation from
    # the original new points to the transformed new points
    transform_temp, loss_affine = rotation_translation(pts_train_trans, pts_train)
    rot = transform_temp[:, :3]
    offset = transform_temp[:, 3]
    # Apply the rigid transformation to the new frame
    fr_trans_rigid = sim.affine_transform(fr, rot, offset, mode='constant', cval=0, order=3)
    # Apply the rigid transformation to the output mask too
    mask_trans_rigid = sim.affine_transform(mask, rot, offset, mode='constant', cval=0, order=0)


    # Figure out the alignment between the z-levels in the two frames
    z_start, z_end, z_target_minus_trans = align_zs(mask_trans_rigid, target_pred_mask_i)

    # Apply optical flow separately to each 2D plane (fixing z)
    # Initialize the warped frame and mask. For z's where there are no target points the points won't be
    # moved.
    frame_warped = fr_trans_rigid.copy()
    mask_warped = mask_trans_rigid.copy()
    z_end_f = np.min([z_end+1,np.shape(mask_trans_rigid)[2],int(31+z_target_minus_trans)])
    for z_idx in range(z_start, z_end_f):
        # Compute the optical flow
        z_idx_trans = int(z_idx - z_target_minus_trans)
        #changed warping parameters to make the deformation more visible
        v, u = optical_flow_tvl1(target_fr_i[:, :, z_idx], fr_trans_rigid[:, :, z_idx_trans], attachment=400, tightness=0.3, num_warp=100)
        nr, nc = target_pred_mask_i[:, :, 0].shape

        # Warp the new image and its mask using the optical flow results
        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
        mask_warped[:, :, z_idx_trans] = warp(mask_trans_rigid[:, :, z_idx_trans],
                                                          np.array([row_coords + v, col_coords + u]),
                                                          mode='nearest', preserve_range=True, order=0)
        frame_warped[:, :, z_idx_trans] = warp(fr_trans_rigid[:, :, z_idx_trans],
                                                        np.array([row_coords + v, col_coords + u]),
                                                        mode='nearest', preserve_range=True, order=0)


    # Ravel the z axis so the z's for the training image align with the z's of the target image
    mask_warped = np.roll(mask_warped, int(z_target_minus_trans), axis=2)
    frame_warped = np.roll(frame_warped, int(z_target_minus_trans), axis=2)

    # Smooth the labels using alpha shapes
    ndim = 2
    if ndim == 2:
        for z in range(mask_warped.shape[-1]):
            all_labels = sorted(np.unique(mask_warped[:, :, z]))[1:]
            for label in all_labels:
                label_pts = np.where(mask_warped[:, :, z] == label)
                label_pts = np.array(label_pts).T
                if len(label_pts) > 2:
                    # Generate the alpha shape
                    for init_denom in range(5, 100, 5):
                        try:
                            alpha_shape = alphashape.alphashape(label_pts, 1/init_denom)
                            # Construct a 2D mesh
                            x = np.arange(0, mask_warped.shape[0])
                            y = np.arange(0, mask_warped.shape[1])
                            points = np.meshgrid(x, y)
                            points = np.array(list(zip(*(dim.flat for dim in points))))
                            # Check if each point falls inside the alpha shape
                            in_shape = [alpha_shape.contains(Point(points[i])) for i in range(len(points))]
                            valid_points = points[in_shape]
                            mask_warped[valid_points[:, 0], valid_points[:, 1], z] = label
                            break
                        except:
                            pass
    else:
        all_labels = sorted(np.unique(mask_warped))[1:]
        for label in all_labels:
            label_pts = np.where(mask_warped == label)
            label_pts = np.array(label_pts).T*scale
            if len(label_pts) > 3:
                alpha_shape = alphashape.alphashape(label_pts, 1/3.0)
                # Convert back to the original scale
                alpha_shape = alpha_shape.apply_scale(1/np.array(scale))
                # Construct a 3D mesh
                x = np.arange(0, mask_warped.shape[0])
                y = np.arange(0, mask_warped.shape[1])
                z = np.arange(0, mask_warped.shape[2])
                points = np.meshgrid(x, y, z)
                points = np.array(list(zip(*(dim.flat for dim in points))))
                # Check if each point falls inside the alpha shape
                try:
                    in_shape = alpha_shape.contains(points)
                except:
                    in_shape = [alpha_shape.contains(Point(points[i])) for i in range(len(points))]
                valid_points = points[in_shape]
                mask_warped[valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]] = label


    frame_warped = frame_warped[np.newaxis, :, :, :]
    return frame_warped, mask_warped,fr_trans_rigid,mask_trans_rigid
