import numpy as np
from matplotlib import cm
import matplotlib as mpl
import scipy.ndimage as sim


# This is a helper class to take the image computation load off the MainFigWidget and controller
class ImageRendering:
    """
    Class used to render the video frames and masks properly and dynamically.
    """
    dimensions = (.1625, .1625, 1.5)  # the extents of a pixel along the x, y and z dimensions

    # AD These are the constant transparencies for mask and mask of highlighted neurons
    mask_transparency = 0.3
    highlighted_transparency = 0.8
    # SJR: default number of mask colors; if there are more neurons than this number, repeat colors
    nmaskcolors = 15#MB changed the color to 15

    def __init__(self, controller, figure, data_name, nb_frames):
        """
        :param figure: instance of MainFigWidget, the figure that displays the image computed
        :param data_name: name of dataset to display in label
        :param nb_frames: total nb of frames in datatset, to display in label
        """
        controller.frame_registered_clients.append(self)
        controller.highlighted_neuron_registered_clients.append(self)
        controller.frame_img_registered_clients.append(self)
        controller.mask_registered_clients.append(self)
        self.figure = figure

        # skeleton of label displayed on the figure.
        # The two labels are concatenated in the end, but their fields for t and z must be filled separately.
        self.label1 = "Dataset: " + data_name + " frame= {}/" + str(nb_frames)
        self.label2 = "; z = {}"

        # The raw red-channel and green-channel images (provided by the controller):
        self.im_rraw = None
        self.im_graw = None
        # The raw mask data (provided by the controller):
        self.raw_mask = None
        # The 3-channel rendered image to be displayed (computed by self):
        self.rendered_img = None
        # The 4-channel rendered mask to be displayed (computed by self):
        self.rendered_mask = None

        # Many image rendering parameters
        self.gamma = 0.4
        self.fast_gamma = True
        self.blend_r = 1
        self.blend_g = 1
        self.high = 100
        self.low = 0
        self.blur_image = False
        self.blur_s = 1
        self.blur_b = 25

        # Initialize mask colors (SJR)
        # SJR: load a color map
        nipy_spectral = cm.get_cmap("nipy_spectral", self.nmaskcolors)
        # SJR: load a set of colors
        cols = nipy_spectral(np.arange(self.nmaskcolors))
        cols[:, -1] = 0.3  # SJR: all objects are largely transparent
        cols[0, -1] = 0  # SJR: object 0 (= background) is completely transparent
        self.cmap_mask = mpl.colors.ListedColormap(
            cols)  # SJR: these are the colors that will be passed on for drawing the mask

        self.highlighted = 0  # which neuron/value in the mask should be highlighted

    def change_t(self, t):
        """Changes the label"""
        label = self.label1.format(t) + self.label2
        self.figure.set_data(label=label)

    def change_autolevels(self):
        self.figure.autolevels = not self.figure.autolevels
        self.figure.update_image_display()

    def change_gamma(self, gamma):
        try:
            self.gamma = int(gamma) / 100
            self._update_image()
        except:
            pass

    def change_blend_r(self, value):
        self.blend_r = value / 100
        self._update_image()

    def change_blend_g(self, value):
        self.blend_g = value / 100
        self._update_image()

    def change_low_thresh(self, value):
        try:
            self.low = value / 100
            self._update_image()
        except:
            pass

    def change_high_thresh(self, value):
        try:
            self.high = value / 100
            self._update_image()
        except:
            pass

    def change_blur_image(self):
        self.blur_image = not self.blur_image
        self._update_image()

    def change_blur_s(self, value):
        self.blur_s = value
        self._update_image()

    def change_blur_b(self, value):
        self.blur_b = value
        self._update_image()

    def change_highlighted_neuron(self, high: int = None, unhigh: int = None, **kwargs):
        """
        Makes the mask of given neuron brighter than the others.
        :param high: neuron id (from 1), will be highlighted if given
        :param unhigh: neuron id (from 1), will be unhighlighted if given
        :param present: NOT USED   # todo: find better name, but I don't know what these colors are for
        """
        if high == self.highlighted:
            return
        if unhigh and high is None:  # only unhighlight currently highlighted
            self.highlighted = 0
        elif high:  # change (or set) highlighted neuron
            self.highlighted = high
        self._update_mask()

    def change_img_data(self, img_r, img_g=None):
        """
        Callback when the video frame displayed changes.
        :param img_r: h*w*d array, the red-channel of the video frame
        :param img_g: h*w*d array, the green-channel of the video frame (or None if only one channel is to be used)
        """
        self.im_rraw = img_r
        self.im_graw = img_g
        self._update_image()

    def change_mask_data(self, mask):
        """
        Callback when the mask displayed changes.
        :param mask: h*w*d array, the mask
        """
        self.raw_mask = mask
        self._update_mask()

    def _f_gamma(self, x):  # CFP
        """Applies fast gamma to the image"""
        exp, w1, w2 = self._find_gamma_pieces(self.gamma)
        if exp < 0:
            for i in range(-exp - 1):
                x = np.sqrt(x)
            return x * w1 + np.sqrt(x) * w2
        for i in range(exp):
            x = x * x
        return x * x * w1 + x * w2

    def _find_gamma_pieces(self, gamma):  # CFP
        """Helper function for _f_gamma"""
        last = 4
        for n, el in zip([1, 0, -1, -2, -3, -4, -5], [2, 1, 0.5, 0.25, 0.125, 0.0625, 0.003125]):
            if gamma >= el:
                w1 = last - gamma
                w2 = gamma - el
                w = w1 + w2
                return n, w2 / w, w1 / w
            last = el

    def compute_rendered_img(self):  # AD
        """
        Computes the image (from the video, independently of the presence of masks or points) to be displayed
        :return: the h*w*3 array of the frame to be displayed
        """
        if self.blur_image:
            # SJR: if blurring chosen, blur the image before doing anything else
            # this needs to be cleaned up, e.g., with respect to dimensions (?).
            # I just copied this from the segmentation code
            sigm = self.blur_s
            bg_factor = self.blur_b
            xysize, xysize2, zsize = self.dimensions
            sdev = np.array([sigm, sigm, sigm * xysize / zsize])
            im = self.im_rraw
            img_r = sim.gaussian_filter(im, sigma=sdev) - sim.gaussian_filter(im, sigma=sdev * bg_factor)
            self.im_rraw = img_r#MB added to change threshold obtained from blurred image
        else:
            img_r = self.im_rraw
        mean_r = np.mean(img_r)

        threshold_r = ((self.low * mean_r) <= img_r)
        img_r = np.clip(threshold_r * img_r, 0, (mean_r + (255 - mean_r) * self.high)) / 255 * self.blend_r
        if self.im_graw is not None:
            mean_g = np.mean(self.im_graw)
            threshold_g = ((self.low * mean_g) <= self.im_graw)
            img_g = np.clip(threshold_g * self.im_graw, 0, (mean_g + (255 - mean_g) * self.high)) / 255 * self.blend_g
        else:
            img_g = img_r * self.blend_g / (self.blend_r + 1e-8)
        img_b = img_r  # blue channel is also green for a two channel image
        # SJR: This is why the red channel is really pink / purple

        # combine the three channels in one
        combined_img = np.concatenate((img_r[:, :, :, None], img_g[:, :, :, None], img_b[:, :, :, None]), axis=3)
        if self.fast_gamma:
            self.rendered_img = self._f_gamma(combined_img)
        else:
            self.rendered_img = combined_img ** self.gamma

    def compute_rendered_mask(self):  # AD
        """
        Computes the mask array to be displayed (transparent where there is no neuron, semi-transparent where there is)
        :return: the h*w*4 array of the masks to be displayed
        """
        # Warning: this could be longer than before because we compute the colormap for the whole mask instead of just
        # one slice (but in return, changing z should be faster)
        mask_rgba = self.cmap_mask((self.raw_mask % self.nmaskcolors + 1) * (self.raw_mask != 0))
        mask_rgba[self.raw_mask == 0, 3] = 0
        mask_rgba[self.raw_mask != 0, 3] = self.mask_transparency

        if self.highlighted > 0:
            mask_rgba[self.raw_mask == self.highlighted, 3] = self.highlighted_transparency

        self.rendered_mask = mask_rgba

    def _update_image(self):
        """
        Recomputes the image rendering and changes the display
        """
        # compute the new image
        self.compute_rendered_img()

        # then actually change the display
        self.figure.set_data(img=self.rendered_img)

    def _update_mask(self):
        """
        Recomputes the mask rendering and changes the display
        """
        if self.raw_mask is None:
            self.figure.set_data(mask=False)
        else:
            # compute the new mask
            self.compute_rendered_mask()

            # then actually change the display
            self.figure.set_data(mask=self.rendered_mask)
