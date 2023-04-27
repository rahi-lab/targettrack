import numpy as np
import scipy.spatial as spat
from tqdm import tqdm
import warnings


class HarvardLab:
    def __init__(self, controller, dataset, settings):
        self.controller = controller
        self.controller.nb_neuron_registered_clients.append(self)
        self.settings=settings

        self.channel_num=dataset.nb_channels
        sh= dataset.frame_shape

        #this is ther kernel calculating the CI
        self.ker_sizexy=int(self.settings["calcium_intensity_kernel_xy"])#plus 1 for the errors
        self.ker_sizez=int(self.settings["calcium_intensity_kernel_z"])
        self.cikernelnorm=(2*self.ker_sizez+1)*(2*self.ker_sizexy+1)**2#pure kernelsize

        self.ker_sizexy+=1
        self.calcium_intensity_fullkernel=np.array(np.meshgrid(np.arange(-self.ker_sizexy,self.ker_sizexy+1),np.arange(-self.ker_sizexy,self.ker_sizexy+1),np.arange(-self.ker_sizez,self.ker_sizez+1),indexing="ij"))
        self.calcium_intensity_kernel_selectors=np.zeros((5,2*self.ker_sizexy+1,2*self.ker_sizexy+1,2*self.ker_sizez+1))

        self.x_min,self.x_max=self.ker_sizexy,sh[0]-self.ker_sizexy-1
        self.y_min,self.y_max=self.ker_sizexy,sh[1]-self.ker_sizexy-1
        self.z_min,self.z_max=self.ker_sizez,sh[2]-self.ker_sizez-1

        self.calcium_intensity_kernel_selectors[0][1:-1,1:-1,:]=1
        self.calcium_intensity_kernel_selectors[1][2:,1:-1,:]=1
        self.calcium_intensity_kernel_selectors[2][1:-1,2:,:]=1
        self.calcium_intensity_kernel_selectors[3][:-2,1:-1,:]=1
        self.calcium_intensity_kernel_selectors[4][1:-1,:-2,:]=1

        self.calcium_intensity_fullkernel=self.calcium_intensity_fullkernel.reshape(3,-1)
        self.calcium_intensity_kernel_selectors=self.calcium_intensity_kernel_selectors.reshape(5,-1)
        assert all(np.sum(self.calcium_intensity_kernel_selectors,axis=1)==self.cikernelnorm), "bug cfpark00@gmail.com"

        ci_int = dataset.ca_act
        if ci_int is None:
            self.ci_int = np.full((self.controller.n_neurons, self.controller.frame_num, 2), np.nan, dtype=np.float32)
        else:
            self.ci_int = ci_int.astype(np.float32)

    def update_ci(self, dataset, t=None, i_from1=None):
        if dataset.point_data:
            self._update_ci_from_pointdat(dataset, t, i_from1)
        else:
            self._update_ci_from_masks(dataset, t, i_from1)

    def _update_single_ci_from_poindat(self, i, ind, loc, valid, im_r, im_g):
        """
        valid is one of None (validity not tested), True (neuron exists and is valid), False (neuron does not exist or
        is not valid).
        """
        if loc is None or valid is False or any(np.isnan(loc)):
            self.ci_int[ind-1][i][0]=np.nan
            self.ci_int[ind-1][i][1]=np.nan
            return

        loc = loc.astype(np.int32)
        if not valid:   # valid is None
            valid=(loc[0]>=self.x_min)*(loc[0]<=self.x_max)*(loc[1]>=self.y_min)*(loc[1]<=self.y_max)*(loc[2]>=self.z_min)*(loc[2]<=self.z_max)
            if not valid:   # now True or False
                self.ci_int[ind - 1][i] = np.nan
                return
        
        if self.channel_num==2:
            loc_ker=loc[:,None]+self.calcium_intensity_fullkernel
            int_gs=np.sum(self.calcium_intensity_kernel_selectors*(im_g[loc_ker[0],loc_ker[1],loc_ker[2]][None,:]),axis=1)/self.cikernelnorm
            int_rs=np.sum(self.calcium_intensity_kernel_selectors*(im_r[loc_ker[0],loc_ker[1],loc_ker[2]][None,:]),axis=1)/self.cikernelnorm
            self.ci_int[ind-1][i][2:7]=int_rs
            self.ci_int[ind-1][i][7:]=int_gs
            ci_int_sing=int_gs/(int_rs+1e-8)
            self.ci_int[ind-1][i][0]=ci_int_sing[0]
            if np.sum(np.isnan(ci_int_sing)==1)>1:#if more than 1 is nan
                self.ci_int[ind-1][i][1]=np.nan
            else:
                self.ci_int[ind-1][i][1]=np.nanstd(ci_int_sing)#elsewise we take it
        else:
            loc_ker=loc[:,None]+self.calcium_intensity_fullkernel
            int_rs=np.sum(self.calcium_intensity_kernel_selectors*(im_r[loc_ker[0],loc_ker[1],loc_ker[2]][None,:]),axis=1)/self.cikernelnorm
            ci_int_sing=int_rs/255
            self.ci_int[ind-1][i][0]=ci_int_sing[0]
            if np.sum(np.isnan(ci_int_sing)==1)>1:
                self.ci_int[ind-1][i][1]=np.nan
            else:
                self.ci_int[ind-1][i][1]=np.nanstd(ci_int_sing)

    def _update_single_ci_from_mask(self, i, ind, present, mask, im_g):
        if not present:
            self.ci_int[ind-1][i] = np.nan
            return
        neuron_values = im_g[mask]
        sorted_values = np.sort(neuron_values)
        n_third = len(sorted_values) // 3
        self.ci_int[ind - 1][i][0] = np.nanmean(sorted_values[-n_third:])
        self.ci_int[ind - 1][i][1] = np.nan

    def _update_ci_from_pointdat(self, dataset, t, i_from1):
        if t is not None:
            times = [t]
        else:
            times = list(range(dataset.frame_num))

        pointdat = self.controller.pointdat

        for i in tqdm(times):
            if all(np.isnan(pointdat[i][:,0])):
                self.ci_int[:, i] = np.nan
                continue

            im_r = dataset.get_frame(i)
            if self.channel_num == 2:
                im_g = dataset.get_frame(i, col="green")
            else:
                im_g = None
            if i_from1 is not None:
                self._update_single_ci_from_poindat(i, i_from1, pointdat[i, i_from1], valid=None, im_r=im_r, im_g=im_g)
            else:
                locs = pointdat[i].astype(np.int32)
                valids = (locs[:, 0] >= self.x_min) * (locs[:, 0] <= self.x_max) * (locs[:, 1] >= self.y_min) * (
                        locs[:, 1] <= self.y_max) * (locs[:, 2] >= self.z_min) * (locs[:, 2] <= self.z_max) * (
                             np.logical_not(np.isnan(np.sum(pointdat[i], axis=1))))
                for j, (loc, valid) in enumerate(zip(locs, valids)):
                    self._update_single_ci_from_poindat(i, j, loc, valid=valid, im_r=im_r, im_g=im_g)

        print()
        print("Done.")

    def _update_ci_from_masks(self, dataset, t, i_from1):
        """
        This computes the activity as was done by EPFL lab.
        """
        if t is not None:
            times = [t]
        else:
            times = list(range(dataset.frame_num))
        for i in tqdm(times):
            mask = dataset.get_mask(i, force_original=True)
            if mask is False:
                self.ci_int[:, i] = np.nan
                continue
            if self.channel_num == 2:
                im_g = dataset.get_frame(i, col="green", force_original=True)
            else:
                im_g = dataset.get_frame(i, col="red", force_original=True)
                warnings.warn("Only one channel, computing intensity from red values.")
            if i_from1 is not None:
                self._update_single_ci_from_mask(i, i_from1, i_from1 in mask, mask=(mask == i_from1), im_g=im_g)
            else:
                self.ci_int[:, i] = np.nan
                for neu in self.controller.present_neurons_at_time(i):
                    self._update_single_ci_from_mask(i, neu, True, mask=(mask == neu), im_g=im_g)
        print()
        print("Done.")

    def change_nb_neurons(self, nb_neurons):
        current_nb_neurons = self.ci_int.shape[0]
        current_ci = self.ci_int
        if current_nb_neurons < nb_neurons:
            self.ci_int = np.full((nb_neurons, self.controller.frame_num, 2), np.nan, dtype=np.float32)
            self.ci_int[:current_nb_neurons] = current_ci
        else:
            self.ci_int = self.ci_int[:nb_neurons]
