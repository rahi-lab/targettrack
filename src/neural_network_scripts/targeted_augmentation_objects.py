import alphashape
import h5py
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage
from shapely.geometry import Point
import shutil
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1
import torch

import NNtools


def load_data(datadir, movie_file, temp_add=0):
    h5 = h5py.File(os.path.join(datadir, movie_file), "r+")
    W, H, D = h5.attrs["W"], h5.attrs["H"], h5.attrs["D"]
    shape = (2, W, H, D)
    allset = NNtools.TrainDataset(datadir, shape)
    NNname = list(h5['net'].keys())
    traininds = h5['net'][NNname[0]].attrs['traininds']

    dataset_name = movie_file.split(".")[0]
    props = dataset_name.split("_")
    runname = props[-1]
    NetName = props[-2]
    identifier = "net/" + NetName + "_" + runname
    num_classes = h5.attrs["N_neurons"] + 1
    NetMod = importlib.import_module(NetName)
    net = NetMod.Net(n_channels=shape[0], num_classes=num_classes)

    T = int(h5.attrs["T"]) - temp_add  # TODO

    return h5, allset, traininds, identifier, net, T, shape, num_classes


def copy_files(datadir, allset, traininds):
    # copy the existing masks first
    dir_deformations = os.path.join(datadir, "deformations")
    if os.path.exists(dir_deformations):
        shutil.rmtree(dir_deformations)  # remove the deformed frames from the previous runs
    os.mkdir(dir_deformations)
    os.mkdir(os.path.join(datadir, "deformations", "frames"))
    os.mkdir(os.path.join(datadir, "deformations", "masks"))
    for name in np.array(allset.filelist)[
        allset.real_ind_to_dset_ind(traininds)]:  # placing the training frames and masks in the deformation folder
        i = int(os.path.split(name)[-1].split(".")[0].split("_")[1])
        shutil.copyfile(os.path.join(datadir, "frames", "frame_" + str(i) + ".npy"),
                        os.path.join(datadir, "deformations", "frames", "frame_" + str(i) + ".npy"))
        shutil.copyfile(name, os.path.join(datadir, "deformations", "masks", "mask_" + str(i) + ".npy"))


def load_target_frame(h5, evalset, i, NNname):
    # Load the new target frame. The mask (second output) may be empty.
    target_fr_i, _ = evalset[i]  # new target frame
    target_fr_i = target_fr_i.squeeze().cpu().detach().numpy()

    knn = "net/" + NNname[0] + "/" + str(i) + "/predmask"
    pred = h5[knn]  # prediction of the target frame
    predmask_i = torch.tensor(pred)
    predmask_i = predmask_i.cpu().detach().numpy().astype(np.int16)  # prediction of the frame out side of training set

    return target_fr_i, predmask_i


def load_closest_training_frame(distmat, traininds, evalset, i):
    i_parent = traininds[np.argmin(distmat[traininds, i])]  # closest training set frame to frame i
    fr, mask = evalset[i_parent]
    mask = mask.cpu().detach().numpy().astype(np.int16)
    fr = fr.squeeze().cpu().detach().numpy()

    return fr, mask, i_parent


def align_zs(mask_trans_rigid, target_pred_mask_i):
    _, _, nonzero_z_trans_rigid = np.nonzero(mask_trans_rigid)
    _, _, nonzero_z_target = np.nonzero(target_pred_mask_i)
    med_z_trans_rigid = np.median(nonzero_z_trans_rigid)
    med_z_target = np.median(nonzero_z_target)
    z_target_minus_trans = med_z_target - med_z_trans_rigid
    z_start, z_end = np.min(nonzero_z_target), np.max(nonzero_z_target)

    return z_start, z_end, z_target_minus_trans


def plot_all_objects(coords, labels, title, xlim=None, ylim=None, markers=True, legend=False, alphas=None, figsize=None,
                     title_fontsize=12, legend_fontsize=12, legend_pointsize=16, tick_fontsize=12, save_file=None,
                     show_fig=True, ax=None, point_size=None, zeros_alpha=0.1):
    plt.clf()
    if figsize is not None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    elif ax is None:
        fig, ax = plt.subplots(1, 1)
    cmap = plt.cm.get_cmap('tab20')
    cmap_colors = list(cmap.colors)
    cmap_colors = cmap_colors[::2][::-1] + cmap_colors[1::2][::-1]
    if markers:
        markers = ['.', '<', 'o', 'd', '^', '8', 's', 'p', 'P', '*', 'h', 'H', 'X', 'D', 'v', '$f$', '$s$']
    if alphas is None:
        alphas = [1]*(np.max(labels)+1)

    if sorted(np.unique(labels))[0] == 0:
        if markers:
            ax.scatter(coords[:, 0][labels == 0], coords[:, 1][labels == 0], color='black', alpha=zeros_alpha,
                       marker=markers[0], edgecolors='black', zorder=1, label=str(0), s=point_size)
        else:
            ax.scatter(coords[:, 0][labels == 0], coords[:, 1][labels == 0], color='black', alpha=zeros_alpha,
                       edgecolors='black', zorder=1, label=str(0), s=point_size)

    if sorted(np.unique(labels))[0] == 0:
        rest_labels = sorted(np.unique(labels))[1:]
    else:
        rest_labels = sorted(np.unique(labels))
    for label in rest_labels:
        if markers:
            ax.scatter(coords[:, 0][labels == label], coords[:, 1][labels == label],
                       color=cmap_colors[label % len(cmap_colors)], alpha=alphas[label],
                       marker=markers[label % len(markers)], edgecolors=cmap_colors[label % len(cmap_colors)], zorder=1,
                       label=str(label), s=point_size)
        else:
            ax.scatter(coords[:, 0][labels == label], coords[:, 1][labels == label],
                       color=cmap_colors[label % len(cmap_colors)], alpha=alphas[label],
                       edgecolors=cmap_colors[label % len(cmap_colors)], zorder=1, label=str(label), s=point_size)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=tick_fontsize)
    if legend:
        leg = ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=legend_fontsize)
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        for handle in leg.legendHandles:
            handle.set_sizes([legend_pointsize])

    if title:
        ax.set_title(title, fontsize=title_fontsize)
    plt.gca().set_aspect('equal', adjustable='box')
    if save_file:
        plt.tight_layout()
        plt.savefig(save_file)
        plt.close()
    elif show_fig:
        plt.show()


def plot_masks(train_mask, target_pred_mask, save_dir):
    for i, mask in enumerate([train_mask, target_pred_mask]):
        coords = np.array(np.where(mask > 0)).T
        labels = mask[np.where(mask > 0)]
        title = 'Training ground-truth mask' if i == 0 else 'Target predicted mask'
        save_file = os.path.join(save_dir, 'train_mask.svg') if i == 0 else os.path.join(save_dir, 'target_mask.svg')
        if len(labels) > 0:
            plot_all_objects(coords, labels, title, markers=True, figsize=(20, 20), xlim=(0, 300),
                                  ylim=(0, 300), legend=True, title_fontsize=32, legend_fontsize=24,
                                  legend_pointsize=300, tick_fontsize=22, alphas=None, save_file=save_file)


def plot_jv_results(mask, target_pred_mask_i, pts_train, pts_train_trans, pts_target, save_dir):
    for i, (dataset_mask, pts) in enumerate(zip([mask, mask, target_pred_mask_i],
                                                [pts_train, pts_train_trans, pts_target])):
        for j in range(3):
            pts[:, j] = np.clip(pts[:, j], a_min=0, a_max=dataset_mask.shape[j])
        pts = np.round(pts).astype(int)
        if i != 1:  # The labels for the pts_train_trans should be the same as for pts_train
            labels = dataset_mask[pts[:, 0], pts[:, 1], pts[:, 2]]
        if i == 0:
            title = 'Original sampled training points'
            save_file = os.path.join(save_dir, 'train_points_orig.svg')
        elif i == 1:
            title = 'JV-transformed sampled training points'
            save_file = os.path.join(save_dir, 'train_points_transformed_JV.svg')
        else:
            title = 'Sampled target points'
            save_file = os.path.join(save_dir, 'target_points.svg')

        plot_all_objects(pts, labels, title, markers=True, figsize=(20, 20), xlim=(0, 300),
                                  ylim=(0, 300), legend=True, title_fontsize=32, legend_fontsize=24,
                                  legend_pointsize=300, tick_fontsize=22, alphas=None, point_size=200, zeros_alpha=0.5,
                                  save_file=save_file)


def plot_rigid_trans_results(fr, fr_trans_rigid, mask_trans_rigid, save_dir):
    # Plot the training frame before and after the transformation
    for i, frame in enumerate([fr, fr_trans_rigid]):
        plt.clf()
        plt.figure(figsize=(20, 20))
        plt.imshow(np.flipud(np.rot90(frame.max(axis=-1))), origin='lower')
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        if i == 0:
            plt.title('Original training frame', fontsize=32)
            save_file = os.path.join(save_dir, 'train_frame.svg')
        else:
            plt.title('Rigidly-transformed training frame', fontsize=32)
            save_file = os.path.join(save_dir, 'train_frame_rigid_trans.svg')
        plt.savefig(save_file)
        plt.close()

    # Plot the transformed mask
    coords = np.array(np.where(mask_trans_rigid > 0)).T
    labels = mask_trans_rigid[np.where(mask_trans_rigid > 0)]
    title = 'Rigidly-transformed training ground-truth mask'
    save_file = os.path.join(save_dir, 'train_mask_rigid_trans.svg')
    plot_all_objects(coords, labels, title, markers=True, figsize=(20, 20), xlim=(0, 300),
                              ylim=(0, 300), legend=True, title_fontsize=32, legend_fontsize=24,
                              legend_pointsize=300, tick_fontsize=22, alphas=None, save_file=save_file)


def plot_optical_flow_results_z(target_fr_i, target_pred_mask_i, fr_trans_rigid, mask_trans_rigid, image1_warp,
                                mask_warp, z_idx, z_idx_trans, save_dir):
    # Plot the target frame at z-level i along with the training frame before and after warping
    for i, frame in enumerate([target_fr_i, fr_trans_rigid, image1_warp]):
        plt.clf()
        plt.figure(figsize=(20, 20))
        if i == 0:
            plt.imshow(np.flipud(np.rot90(frame[:, :, z_idx])), origin='lower')
        else:
            plt.imshow(np.flipud(np.rot90(frame[:, :, z_idx_trans])), origin='lower')
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        if i == 0:
            plt.title('Target frame at z = ' + str(z_idx), fontsize=32)
            save_file = os.path.join(save_dir, 'target_frame_z' + str(z_idx) + '.svg')
        elif i == 1:
            plt.title('Rigidly-transformed training frame at z = ' + str(z_idx), fontsize=32)
            save_file = os.path.join(save_dir, 'train_frame_rigid_trans_z' + str(z_idx) + '.svg')
        else:
            plt.title('Optical-flow transformed training frame at z = ' + str(z_idx), fontsize=32)
            save_file = os.path.join(save_dir, 'train_frame_optical_flow_z' + str(z_idx) + '.svg')

        plt.savefig(save_file)
        plt.close()

    # Plot the target mask at z-level i along with the training mask before and after warping
    for i, mask in enumerate([target_pred_mask_i, mask_trans_rigid, mask_warp]):
        if i == 0:
            coords = np.array(np.where(mask[:, :, z_idx] > 0)).T
        else:
            coords = np.array(np.where(mask[:, :, z_idx_trans] > 0)).T
        if len(coords) > 0:
            if i == 0:
                labels = mask[np.where(mask[:, :, z_idx] > 0)][:, z_idx]
            else:
                labels = mask[np.where(mask[:, :, z_idx_trans] > 0)][:, z_idx_trans]
            if i == 0:
                title = 'Target mask at z = ' + str(z_idx)
                save_file = os.path.join(save_dir, 'target_mask_z' + str(z_idx) + '.svg')
            elif i == 1:
                labels = labels.astype(int)
                title = 'Rigidly-transformed training mask at z = ' + str(z_idx)
                save_file = os.path.join(save_dir, 'train_mask_rigid_trans_z' + str(z_idx) + '.svg')
            else:
                labels = labels.astype(int)
                title = 'Optical-flow transformed training mask at z = ' + str(z_idx)
                save_file = os.path.join(save_dir, 'train_mask_optical_flow_z' + str(z_idx) + '.svg')

            plot_all_objects(coords, labels, title, markers=True, figsize=(20, 20), xlim=(0, 300),
                                      ylim=(0, 300), legend=True, title_fontsize=32, legend_fontsize=24,
                                      legend_pointsize=300, tick_fontsize=22, alphas=None, save_file=save_file)


def plot_optical_flow_results_projected(pts_train_warped, mask_warped, save_dir):
    # Plot the projected training frame post-warping
    plt.clf()
    plt.figure(figsize=(20, 20))
    plt.imshow(np.flipud(np.rot90(np.max(pts_train_warped, axis=-1))), origin='lower')
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.title('Optical-flow transformed training frame', fontsize=32)
    save_file = os.path.join(save_dir, 'train_frame_optical_flow.svg')
    plt.savefig(save_file)
    plt.close()

    # Plot the projected training mask post-warping
    coords = np.array(np.where(mask_warped > 0)).T
    labels = mask_warped[np.where(mask_warped > 0)]
    title = 'Optical-flow transformed training mask'
    save_file = os.path.join(save_dir, 'train_mask_optical_flow.svg')
    plot_all_objects(coords, labels, title, markers=True, figsize=(20, 20), xlim=(0, 300),
                              ylim=(0, 300), legend=True, title_fontsize=32, legend_fontsize=24,
                              legend_pointsize=300, tick_fontsize=22, alphas=None, save_file=save_file)


def plot_alpha_shape_results(mask_warped, save_dir):
    # Plot the projected training mask post-warping
    plt.clf()
    coords = np.array(np.where(mask_warped > 0)).T
    labels = mask_warped[np.where(mask_warped > 0)]
    title = 'Transformed training mask with alpha shapes'
    save_file = os.path.join(save_dir, 'train_mask_alpha_shape.svg')
    plot_all_objects(coords, labels, title, markers=True, figsize=(20, 20), xlim=(0, 300),
                              ylim=(0, 300), legend=True, title_fontsize=32, legend_fontsize=24,
                              legend_pointsize=300, tick_fontsize=22, alphas=None, save_file=save_file)
    plt.close()

    # Plot the training mask for each z post-warping
    for i in range(mask_warped.shape[-1]):
        plt.clf()
        coords = np.array(np.where(mask_warped[:, :, i] > 0)).T
        z_idxs = np.where(mask_warped[:, :, i] > 0)
        labels = mask_warped[z_idxs[0], z_idxs[1], i]
        if len(coords) > 0:
            title = 'Transformed training mask with alpha shapes, z = ' + str(i)
            save_file = os.path.join(save_dir, 'train_mask_alpha_shape' + '_z' + str(i) + '.svg')
            plot_all_objects(coords, labels, title, markers=True, figsize=(20, 20), xlim=(0, 300),
                                      ylim=(0, 300), legend=True, title_fontsize=32, legend_fontsize=24,
                                      legend_pointsize=300, tick_fontsize=22, alphas=None, save_file=save_file)
            plt.close()


def targeted_augmentation(h5, num_additional, datadir, allset, traininds, T, identifier, shape, num_classes,
                          scale=(0.1625, 0.1625, 1.5), plot_results=True, plots_dir=None):
    copy_files(datadir, allset, traininds)
    if plots_dir and not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    distmat = np.array(h5["distmat"])
    additional_inds = NNtools.select_additional(T, traininds, distmat, num_additional)[len(traininds):]  # index of frames used for augmentation
    h5[identifier].attrs["Deforminds"] = additional_inds  # MB added
    evalset = NNtools.EvalDataset(datadir, shape)  # gets the new frame(out side of training set) .npy files
    ExtframeCount = 0
    NNname = list(h5['net'].keys())
    totFrameChecked=0
    while num_additional > 0:
        totFrameChecked = totFrameChecked + num_additional
        num_additional = 0

        for i in additional_inds:
            # Load the target frame and the current predicted mask
            target_fr_i, target_pred_mask_i = load_target_frame(h5, evalset, i, NNname)

            # Check whether the mask has at least num_classes-2 objects. If not, use a different frame.
            if len(np.unique(target_pred_mask_i)) < (num_classes - 2):
                num_additional = num_additional + 1
                print("frame not accepted")
            else:
                # Load the training frame that is closest to the given target frame, along with its corresponding
                # ground-truth mask
                fr, mask, i_parent = load_closest_training_frame(distmat, traininds, evalset, i)
                print('Warping frame', i_parent, 'to', i)

                # Plot the training mask and estimated target mask to examine later
                if plot_results:
                    plots_dir_i = os.path.join(plots_dir, str(i))
                    if not os.path.exists(plots_dir_i):
                        os.makedirs(plots_dir_i)
                    plot_masks(mask, target_pred_mask_i, plots_dir_i)

                # Perform Jian-Vemuri point-cloud registration to estimate a non-rigid (thin-plate spline) mapping from
                # stardardized points in the contours of objects in the training mask to standardized points in the
                # contours of objects in the target mask.
                # pts_train: the training set points that we are transforming
                # pts_train_trans: training set points after transformation
                # pts_target: goal of the transformation which is the predicted mask
                pts_train, pts_train_trans, pts_target = NNtools.get_pts_from_masksJV(mask, target_pred_mask_i)

                # TODO I don't think the code below is needed, but I'm leaving it just in case...
                NanElements = np.argwhere(np.isnan(pts_train_trans[:, 1]))
                if len(NanElements) > 0:
                    raise ValueError
                    pts_train = np.delete(pts_train, NanElements, 0)
                    pts_train_trans = np.delete(pts_train_trans, NanElements, 0)

                # Plot the results from the J-V registration
                if plot_results:
                    plot_jv_results(mask, target_pred_mask_i, pts_train, pts_train_trans, pts_target, plots_dir_i)

                # Compute the 2d translation + rotation matrix that best approximates the non-rigid transformation from
                # the original new points to the transformed new points
                transform_temp, loss_affine = NNtools.rotation_translation(pts_train_trans, pts_train)
                rot = transform_temp[:, :3]
                offset = transform_temp[:, 3]
                # Apply the rigid transformation to the new frame
                fr_trans_rigid = scipy.ndimage.affine_transform(fr, rot, offset, mode='constant', cval=0, order=3)
                # Apply the rigid transformation to the output mask too
                mask_trans_rigid = scipy.ndimage.affine_transform(mask, rot, offset, mode='constant', cval=0, order=0)

                # Plot the rigid transformation results
                if plot_results:
                    plot_rigid_trans_results(fr, fr_trans_rigid, mask_trans_rigid, plots_dir_i)

                # Figure out the alignment between the z-levels in the two frames
                z_start, z_end, z_target_minus_trans = align_zs(mask_trans_rigid, target_pred_mask_i)

                # Apply optical flow separately to each 2D plane (fixing z)
                # Initialize the warped frame and mask. For z's where there are no target points the points won't be
                # moved.
                pts_train_warped = fr_trans_rigid.copy()
                mask_warped = mask_trans_rigid.copy()
                z_end_f = np.min([z_end+1,np.shape(mask_trans_rigid)[2],int(31+z_target_minus_trans)])
                for z_idx in range(z_start, z_end_f):
                    # Compute the optical flow
                    z_idx_trans = int(z_idx - z_target_minus_trans)
                    v, u = optical_flow_tvl1(target_fr_i[:, :, z_idx], fr_trans_rigid[:, :, z_idx_trans])
                    nr, nc = target_pred_mask_i[:, :, 0].shape

                    # Warp the new image and its mask using the optical flow results
                    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
                    mask_warped[:, :, z_idx_trans] = warp(mask_trans_rigid[:, :, z_idx_trans],
                                                          np.array([row_coords + v, col_coords + u]),
                                                          mode='nearest', preserve_range=True, order=0)
                    pts_train_warped[:, :, z_idx_trans] = warp(fr_trans_rigid[:, :, z_idx_trans],
                                                        np.array([row_coords + v, col_coords + u]),
                                                        mode='nearest', preserve_range=True, order=0)

                    # Plot the optical flow results for this z
                    if plot_results:
                        plot_optical_flow_results_z(target_fr_i, target_pred_mask_i, fr_trans_rigid, mask_trans_rigid,
                                                    pts_train_warped, mask_warped, z_idx, z_idx_trans, plots_dir_i)

                # Ravel the z axis so the z's for the training image align with the z's of the target image
                mask_warped = np.roll(mask_warped, int(z_target_minus_trans), axis=2)
                pts_train_warped = np.roll(pts_train_warped, int(z_target_minus_trans), axis=2)

                # Plot the projected optical flow results
                plot_optical_flow_results_projected(pts_train_warped, mask_warped, plots_dir_i)

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

                # Plot the results for each z level
                if plot_results:
                    plot_alpha_shape_results(mask_warped, plots_dir_i)

                pts_train_warped = pts_train_warped[np.newaxis, :, :, :]
                # TODO I'm not sure whether or not the line below is needed, but I think it isn't
                # pts_train_warped = np.clip(pts_train_warped * 255, 0, 255)

                # Save the results
                # Compute the frame index for the generated frame. We add T to avoid collision, but don't want to define
                # another class.
                #temp_add=0#TODO MB
                frame_idx = str(T + ExtframeCount)
                # Save the frame and mask
                print("frameind")
                print(frame_idx)
                np.save(os.path.join(datadir, "deformations", "frames", "frame_" + frame_idx + ".npy"),
                        pts_train_warped)
                np.save(os.path.join(datadir, "deformations", "masks", "mask_" + frame_idx + ".npy"), mask_warped)
                # Add the frame and mask to the dataset
                pts_train_warped[0]= np.clip(pts_train_warped[0]*255,0,255)
                h5.attrs["oldT"]=T
                dset = h5.create_dataset(frame_idx + "/frame", pts_train_warped.shape, dtype="i2", compression="gzip")
                dset[...] = pts_train_warped
                dset = h5.create_dataset(frame_idx + "/mask", mask_warped.shape, dtype="i2", compression="gzip")
                dset[...] = mask_warped
                # Update the number of frames
                h5.attrs["T"] = int(frame_idx)
                # Update the number of extra frames
                ExtframeCount += 1
        if num_additional > 0 :# if not enough frames were added
            additional_inds=NNtools.select_additional(T,traininds,distmat,totFrameChecked + num_additional)[len(traininds)+totFrameChecked:]
