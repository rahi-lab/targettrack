import cv2
import numpy as np
import scipy.ndimage as sim


# todo: This file is in mask_processing because its functions are only used in the case of mask data, but it applies to
#  video frames as well as masks. Could it be useful for point data?


def blur(frame, blur_b=40, blur_s=6, Subt_bg=False, subtVal=1):
    """this blurs the images and subtracts background if asked"""
    dimensions = (.1625, .1625, 1.5)
    sigm = blur_s  # value between 1-10
    bg_factor = blur_b  # value between 0-100
    xysize, xysize2, zsize = dimensions
    sdev = np.array([sigm, sigm, sigm * xysize / zsize])
    im_rraw = frame
    im = im_rraw
    sm = sim.gaussian_filter(im, sigma=sdev) - sim.gaussian_filter(im, sigma=sdev * bg_factor)
    im_rraw = sm
    if Subt_bg:
        threshold_r = (im_rraw < subtVal)
        im_rraw[threshold_r] = 0

    frame = im_rraw

    return frame


def SubtBg(frame, subtVal):
    # Todo: possibly rename function and its parameters
    """
    subtracts a constant background value from your movie   # TODO MB looks like it is rather setting to black the background as defined by a threshold (if you agree we should make it clear in the docstring, and possibly rename the function and its parameters)
    """
    im_rraw = frame
    threshold_r=(im_rraw<subtVal)
    im_rraw[threshold_r]=0
    return im_rraw


def resize_frame(frame, width,height, mask=False):
    """resizes the frame to the dimensions given as width and height"""
    frameResize = np.zeros((width,height,np.shape(frame)[2]))
    for j in range(np.shape(frame)[2]):
        if mask:
            frameResize[:,:,j] = cv2.resize(frame[:,:,j], dsize=(height, width), interpolation=cv2.INTER_NEAREST)
        else:
            frameResize[:,:,j] = cv2.resize(frame[:,:,j], dsize=(height, width), interpolation=cv2.INTER_CUBIC)
    return frameResize

