import numpy as np
from scipy import ndimage
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt
from ..helpers.helpers import project


class ImageAligner:
    """Applies the affine transformations defined in self.data"""
    def __init__(self, data):
        self.data = data   # an instance of DataSet

    def align(self, image, t, ismask=False):
        """
        Applies to image the transformation defined in self.data so as to align the frame to a reference.
        :param image: np array, the image (possibly a mask) to align
        :param t: Integer, time frame the image comes from
        :param ismask: ismask if the image is a mask
        :return : The new rotated and translated image.
        """
        img_frame = image
        transform = self.data.get_transformation(t)
        if transform is None:
            return img_frame

        if ismask:
            order = 0
            cval = 0
        else:
            order = 3   # default value of affine_transform
            cval = np.median(img_frame)  # why median??
        new_image = self.apply_transform(img_frame, transform, mode='constant', cval=cval, order=order)
        return new_image

    def dealign(self, image, t,centerRot):
        """
        Rotate the image back to its original position, inverting the rotation and translating.
        :param image: nparray
        :param t: Integer, time frame t
        :param centerRot: 0 or 1. which python function is used for transformation
        :return: nparray, the de-aligned image
        """
        img_frame = image
        if centerRot==0:
            transform = self.data.get_transformation(t)
            # Image is always a mask in our use cases
            new_image = self.apply_inverse_transform(img_frame, transform, mode='constant', cval=0, order=0)
        else:
            Angle,offset = self.data.get_transfoAngle(t)
            new_image = self.apply_inverse_transform(img_frame, transform=0, mode='constant', cval=0, order=0,centerRot=1,angleDeg=-Angle,offset=-offset)
        return new_image

    def apply_transform(self,image,transform, mode = 'wrap',cval = 0, order=3):
        """
        Apply transform to the image
        For last 3 parameters refer- https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html
        :param image: numpy array, image
        :param transform: numpy array 3 x 4, The first 3 columns contains rotation matrix and the last column contains the translation offset
        :param mode: string, one in ['wrap','constant',...] The mode for the ndimage affine transform see reference above
        :param cval: The constant to be filled in mode constant
        :param order: see reference above
        :return:
        """
        rot = transform[:,:3]
        offset = transform[:,3]
        new_image = affine_transform(image,rot,offset,mode=mode,cval = cval, order=order)
        return new_image

    def apply_inverse_transform(self,image,transform, mode = 'wrap',cval = 0, order=3,centerRot=0,angleDeg=0,offset=0):
        """
        Apply inverse transform to the image
        For last 3 parameters refer- https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html
        :param image: numpy array The image of the
        :param transform: numpy array 4 x 3, The first 3 columns contains rotation matrix and the last column contains the translation offset
        :param mode: string, one in ['wrap','constant',...] The mode for the ndimage affine transform see reference above
        :param cval: The constant to be filled in mode constant
        :param order: see reference above
        :return:
        """
        if centerRot == 1:
            rot1 =  [[1,0,0],[0,1,0],[0,0,1]]
            image = affine_transform(image,rot1,offset,mode=mode,cval = cval, order=order)
            new_image = ndimage.rotate(image, angle=angleDeg, reshape=False,mode=mode,cval = cval, order=order)
        else:
            rot = transform[:,:3]
            offset = transform[:,3]
            rot_inv = np.linalg.inv(rot)
            offset_inv = np.dot(rot_inv,offset)*-1
            new_image = affine_transform(image,rot_inv,offset_inv,mode=mode,cval = cval, order=order)
        return new_image

    def plot_images(self,images,axis_names,fname = None):
        """
        Plots three images, ( original, reference, rotated )
        :param images: an array of 3 images
        :param axis_names: an array of 3 axis names, string
        :param fname: file name to store the output pyplot, will show incase of None.
        """
        fig = plt.figure()
        axs = fig.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 1]})

        print("Plotting Image")
        for i in range(len(images)):
            image = images[i]
            ax = axs[i]
            aspect = "equal"
            origin = "upper"
            ax.imshow(project(image, 2), aspect=aspect, origin=origin)
            ax.set_title(axis_names[i])
        plt.draw()
        if fname is not None:
            plt.savefig(fname)
        else:
            plt.show()


class ImageCropper:
    def __init__(self, data, orig_shape=None):
        self.data = data   # an instance of DataSet
        self.orig_shape = orig_shape

    def crop(self, image):
        """
        Crops the image around the Region Of Interest defined by self.data.
        :param image: image to be resized (could be a mask)
        :return: cropped_frame the cropped image
        """

        image_shape = image.shape
        if self.orig_shape is None:
            self.orig_shape = image_shape  # Todo: always safe??
        assert self.orig_shape == image_shape, "ImageResizerCrop has not been designed to deal with images of different sizes, you will have problems with inverse resizing."
        # Todo: dealing with images of different sizes could be useful if we want to shrink images when rotating??
        x_left, x_right, y_left, y_right = self.data.get_ROI_params()
        x_left, x_right, y_left, y_right = self._find_crop_lims(x_left, x_right, y_left, y_right)

        cropped_frame = image[x_left:x_right, y_left:y_right, :]   # Todo: take care of z properly
        return cropped_frame

    def inverse_crop(self, mask):
        """
        Performs inverse cropping on mask: adds background to fill the space around the ROI defined by self.data.
        :param mask: image to be resized
        :return: new_mask the resized image
        """

        image_shape = mask.shape
        #assert image_shape[2] == 32

        x_left, x_right, y_left, y_right = self.data.get_ROI_params()
        x_left, x_right, y_left, y_right = self._find_crop_lims(x_left, x_right, y_left, y_right)
        new_mask = np.zeros(self.orig_shape)
        new_mask[x_left:x_right, y_left:y_right, :] = mask#MB changed 33 to 1:image_shape[2]+1
        return new_mask

    def _find_crop_lims(self, x_left, x_right, y_left, y_right):
        """
        Finds the limits of the region that will be cropped, given the limits of the Region Of Interest.
        The region that will be cropped includes the entire ROI, but can be slightly larger in order to make the frame
        shape multiple of 16 (for the NN).
        :param x_left, x_right, y_left, y_right: limits of the ROI.
        :return: crop_x_left, crop_x_right, crop_y_left, crop_y_right, the limits of the region to crop
        """
        missing_x = (x_left - x_right) % 32   # number of pixels to be added along x for the width to be multiple of 16
        crop_x_left = max(0, x_left - missing_x)   # add missing pixels to the left, if possible
        crop_x_right = x_right + missing_x - (x_left - crop_x_left)   # add remaining missing pixels to the right (if left space was too short)
        missing_y = (y_left - y_right) % 32   # number of pixels to be added along y for the width to be multiple of 16
        crop_y_left = max(0, y_left - missing_y)   # add missing pixels to the left, if possible
        crop_y_right = y_right + missing_y - (y_left - crop_y_left)   # add remaining missing pixels to the right (if left space was too short)
        return crop_x_left, crop_x_right, crop_y_left, crop_y_right
