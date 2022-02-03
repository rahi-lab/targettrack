import numpy as np
import scipy.spatial as spat
from tqdm import tqdm
import warnings


class HarvardLab:
    def __init__(self,dataset,settings):
        self.settings=settings

        self.frame_num = len(dataset.frames)
        self.channel_num=dataset.nb_channels
        self.sh= dataset.frame_shape
        self.n_neurons=dataset.nb_neurons
        if dataset.point_data:
            self.pointdat=np.array(dataset.pointdat)
        else:
            self.pointdat=np.full((self.frame_num,self.n_neurons,3),np.nan)

        #this is ther kernel calculating the CI
        self.ker_sizexy=int(self.settings["calcium_intensity_kernel_xy"])#plus 1 for the errors
        self.ker_sizez=int(self.settings["calcium_intensity_kernel_z"])
        self.cikernelnorm=(2*self.ker_sizez+1)*(2*self.ker_sizexy+1)**2#pure kernelsize

        self.ker_sizexy+=1
        self.calcium_intensity_fullkernel=np.array(np.meshgrid(np.arange(-self.ker_sizexy,self.ker_sizexy+1),np.arange(-self.ker_sizexy,self.ker_sizexy+1),np.arange(-self.ker_sizez,self.ker_sizez+1),indexing="ij"))
        self.calcium_intensity_kernel_selectors=np.zeros((5,2*self.ker_sizexy+1,2*self.ker_sizexy+1,2*self.ker_sizez+1))

        self.x_min,self.x_max=self.ker_sizexy,self.sh[0]-self.ker_sizexy-1
        self.y_min,self.y_max=self.ker_sizexy,self.sh[1]-self.ker_sizexy-1
        self.z_min,self.z_max=self.ker_sizez,self.sh[2]-self.ker_sizez-1

        self.calcium_intensity_kernel_selectors[0][1:-1,1:-1,:]=1
        self.calcium_intensity_kernel_selectors[1][2:,1:-1,:]=1
        self.calcium_intensity_kernel_selectors[2][1:-1,2:,:]=1
        self.calcium_intensity_kernel_selectors[3][:-2,1:-1,:]=1
        self.calcium_intensity_kernel_selectors[4][1:-1,:-2,:]=1

        self.calcium_intensity_fullkernel=self.calcium_intensity_fullkernel.reshape(3,-1)
        self.calcium_intensity_kernel_selectors=self.calcium_intensity_kernel_selectors.reshape(5,-1)
        assert all(np.sum(self.calcium_intensity_kernel_selectors,axis=1)==self.cikernelnorm), "bug cfpark00@gmail.com"


        #this calculates the ci_ints if not already done
        try:
            self.ci_int = np.array(dataset.ci_int())
        except KeyError:
            self.update_all_ci(dataset)
            self.ci_int=self.ci_int.astype(np.float32)
            dataset.set_calcium(self.ci_int.astype(np.float32))

    #saving the calcium intensities
    def save_ci_int(self, dataset):
        dataset.set_calcium(self.ci_int.astype(np.float32))

    def update_single_ci(self, dataset, i, ind, loc):   # TODO: this (and more?) should be done by callback to change_pointdats or change_mask?
        if dataset.point_data:
            self.update_single_ci_from_poindat(dataset, i, ind, loc)
        else:
            self.update_single_ci_from_mask(dataset, i, ind, loc)

    def update_single_ci_from_poindat(self, dataset, i, ind, loc, validated=False, im_r=None, im_g=None):
        if loc is None:
            self.ci_int[ind-1][i][0]=np.nan
            self.ci_int[ind-1][i][1]=np.nan
            return
        if any(np.isnan(loc)):   # todo: can that happen?
            return   # TODO: return np.nan?
        loc=loc.astype(np.int32)
        if not validated:
            valid=(loc[0]>=self.x_min)*(loc[0]<=self.x_max)*(loc[1]>=self.y_min)*(loc[1]<=self.y_max)*(loc[2]>=self.z_min)*(loc[2]<=self.z_max)
            if not valid:
                return
        if im_r is None:
            im_r = dataset.get_frame(i)
        if self.channel_num==2:
            if im_g is None:
                im_g = dataset.get_frame(i, col="green")
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

    def update_single_ci_from_mask(self, dataset, i, ind, loc, mask=None, im_g=None):
        if loc is None:
            self.ci_int[ind-1][i][0] = np.nan
            self.ci_int[ind-1][i][1] = np.nan
            return
        if any(np.isnan(loc)):
            return
        if im_g is None:
            if self.channel_num==2:
                im_g = dataset.get_frame(i, col="green", force_original=True)
            else:
                im_g = dataset.get_frame(i, col="red", force_original=True)
                warnings.warn("Only one channel, computing intensity from red values.")
        if mask is None:
            mask = dataset.get_mask(i, force_original=True)
        self.ci_int[ind - 1][i][0] = np.sum(im_g[mask])  # TODO: could normalize by red intensity (or take only first decile brightest)...
        self.ci_int[ind - 1][i][1] = np.nan
        #print(np.shape(self.ci_int))#MB check


    def update_pointdat(self,pointdat):
        self.pointdat=pointdat

    #this calculates calcium intensities at all times
    def update_all_ci(self, dataset):
        self.ci_int = np.full((self.n_neurons,self.frame_num,12),np.nan)
        print("Calculating CI intensities.")
        if dataset.point_data:
            self.update_all_ci_from_pointdat(dataset)
        else:
            self.update_all_ci_from_masks(dataset)

    def update_all_ci_from_pointdat(self, dataset):
        for i in tqdm(range(self.frame_num)):
            if all(np.isnan(self.pointdat[i][:,0])):
                continue
            locs = self.pointdat[i].astype(np.int32)
            valids = (locs[:, 0] >= self.x_min) * (locs[:, 0] <= self.x_max) * (locs[:, 1] >= self.y_min) * (
                    locs[:, 1] <= self.y_max) * (locs[:, 2] >= self.z_min) * (locs[:, 2] <= self.z_max) * (
                         np.logical_not(np.isnan(np.sum(self.pointdat[i], axis=1))))

            im_r = dataset.get_frame(i)
            if self.channel_num == 2:
                im_g = dataset.get_frame(i, col="green")
            else:
                im_g = None

            for j, (loc, valid) in enumerate(zip(locs, valids)):
                if not valid:
                    continue
                self.update_single_ci_from_poindat(dataset, i, j, loc, validated=True, im_r=im_r, im_g=im_g)

        print()
        print("Done.")

    def update_all_ci_from_masks(self, dataset):
        """
        This computes the activity as was done by EPFL lab.
        """
        for i in tqdm(range(self.frame_num)):
            try:
                mask = dataset.get_mask(i, force_original=True)
            except KeyError:
                continue
            if self.channel_num == 2:
                im_g = dataset.get_frame(i, col="green", force_original=True)
            else:
                im_g = dataset.get_frame(i, col="red", force_original=True)
                warnings.warn("Only one channel, computing intensity from red values.")
            for neu in np.unique(mask):
                if not neu:   # remove 0
                    continue
                self.update_single_ci_from_mask(dataset, i, neu, [1], mask=(mask == neu), im_g=im_g)
        print()
        print("Done.")

    def get_mask(self,i,dataset,pts,num_classes,thres=4,distthres=4):
        im=np.array(dataset[str(i)+"/frame"])[0]#red channel
        valid=(np.isnan(pts[:,0])!=1)
        pts=pts[valid]
        if len(pts)<4:
            return np.zeros(im.shape).astype(np.int16)
        labs=np.arange(num_classes)[valid]
        tree=spat.cKDTree(pts)
        grid = np.array(np.meshgrid(np.arange(self.sh[0]), np.arange(self.sh[1]),
                                    np.arange(self.sh[2]), indexing="ij")).reshape(3,-1).T
        ds,iis=tree.query(grid,k=1)
        mask=labs[iis].reshape(self.sh)
        return mask*(im>thres)*(ds.reshape(self.sh)<distthres)
