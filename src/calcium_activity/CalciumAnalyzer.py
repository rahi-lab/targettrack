import numpy as np
import scipy.spatial as spat
from tqdm import tqdm
import warnings
from logging_config import setup_logger
logger = setup_logger(__name__)

class CalciumAnalyzer:
    def __init__(        
          self,
          num_neurons,
          num_frames,
          frame_shape,
          settings,
          channel_num=2,  # can be 1 or 2
        ):
        """
        :param ca_act: Old dataset if exists
        :param num_neurons: Number of neurons
        :param num_frames: Number of frames
        :param frame_shape: (X, Y, Z) shape of each 3D frame
        :param settings: Dict-like settings containing kernel sizes, etc.
        :param channel_num: 1 if single-channel, 2 if dual-channel
        """
        
        self.num_neurons = num_neurons
        self.num_frames = num_frames
        self.frame_shape = frame_shape
        self.channel_num = channel_num

        #Kernel sizes
        self.ker_sizexy=int(settings["calcium_intensity_kernel_xy"])
        self.ker_sizez=int(settings["calcium_intensity_kernel_z"])
        self.cikernelnorm=(2*self.ker_sizez+1)*(2*self.ker_sizexy+1)**2#pure kernelsize
        
        # Build the (x, y, z) offsets of the kernel
        
        self.ker_sizexy+=1 #plus 1 for the errors
        self.calcium_intensity_fullkernel=np.array(
            np.meshgrid(
              np.arange(-self.ker_sizexy,self.ker_sizexy+1),
              np.arange(-self.ker_sizexy,self.ker_sizexy+1),
              np.arange(-self.ker_sizez,self.ker_sizez+1),
              indexing="ij"
            )
        )
        self.calcium_intensity_kernel_selectors=np.zeros((
          5,
          2*self.ker_sizexy+1,
          2*self.ker_sizexy+1,
          2*self.ker_sizez+1
            
        ))

        # Compute boundary limits for valid kernel center:
        self.x_min = self.ker_sizexy
        self.x_max = frame_shape[0] - self.ker_sizexy - 1
        self.y_min = self.ker_sizexy
        self.y_max = frame_shape[1] - self.ker_sizexy - 1
        self.z_min = self.ker_sizez
        self.z_max = frame_shape[2] - self.ker_sizez - 1

        self.calcium_intensity_kernel_selectors[0][1:-1,1:-1,:]=1
        self.calcium_intensity_kernel_selectors[1][2:,1:-1,:]=1
        self.calcium_intensity_kernel_selectors[2][1:-1,2:,:]=1
        self.calcium_intensity_kernel_selectors[3][:-2,1:-1,:]=1
        self.calcium_intensity_kernel_selectors[4][1:-1,:-2,:]=1

        self.calcium_intensity_fullkernel=self.calcium_intensity_fullkernel.reshape(3,-1)
        self.calcium_intensity_kernel_selectors=self.calcium_intensity_kernel_selectors.reshape(5,-1)
        assert all(np.sum(self.calcium_intensity_kernel_selectors,axis=1)==self.cikernelnorm), "bug cfpark00@gmail.com"
        
        
        self.ci_int = np.full((num_neurons, num_frames, channel_num), np.nan, dtype=np.float32)
    def update_ci_pointdata(self, pointdat, frame, t=None):
      """
      Updates self.ci_int for all frames and neurons using point-data.

      :param pointdat: shape (num_frames, num_neurons, 3) array of x, y, z coords
      :param frame: 3D volume of the frame of interest
      :param t: The timestep of the frame (to cross check with pointdat)
      """
      if t is not None:
          times = 1
      else:
          times = self.num_frames
      total_changes = np.zeros((times, self.num_neurons, 2))
      for i in range(times):
          # frames_r[i] is the 3D volume for frame i (red channel)
          im_r = frame[0]
          if self.channel_num == 2:
              # frames_g[i] is the 3D volume for frame i (green channel)
              im_g = frame[1]
          else:
              im_g = None
          frame_changes = np.zeros((self.num_neurons, 2))
          for neuron_idx in range(self.num_neurons):
              loc = pointdat[i, neuron_idx]
              # If loc is NaN or out of bounds, skip
              if np.any(np.isnan(loc)):
                  self.ci_int[neuron_idx - 1, i] = np.nan
                  continue
              logger.debug(f"{neuron_idx} neuron actually checked and has location f{loc}")
              frame_changes[i] = self.update_single_ci_from_poindat(
                  frame_i=i,
                  neuron_idx=neuron_idx-1,
                  loc=loc,
                  im_r=im_r,
                  im_g=im_g
                )
          total_changes[i] = frame_changes
              
          return total_changes

    # Look out for an error by 1 with neuron idx
    def update_single_ci_from_poindat(self, frame_i, neuron_idx, loc, im_r, im_g=None):
      # Convert to int for indexing
      loc = loc.astype(np.int32)

      # Check boundary:
      if (loc[0] < self.x_min or loc[0] > self.x_max or
          loc[1] < self.y_min or loc[1] > self.y_max or
          loc[2] < self.z_min or loc[2] > self.z_max):
          self.ci_int[neuron_idx, frame_i] = np.nan
          return None

      # Offsets for kernel
      loc_ker = loc[:, None] + self.calcium_intensity_fullkernel  # shape: (3, N)

      # Extract red intensities in the kernel
      # shape after indexing: (N,)
      intens_r = im_r[loc_ker[0], loc_ker[1], loc_ker[2]]

      # Multiply by each of the 5 sub-kernel selectors, sum, then normalize
      int_rs = np.sum(
          self.calcium_intensity_kernel_selectors * intens_r[None, :],
          axis=1
      ) / self.cikernelnorm  # shape: (5,)
      ci_val = 0
      std = np.nan
      if self.channel_num == 2 and im_g is not None:
          # Extract green intensities
          intens_g = im_g[loc_ker[0], loc_ker[1], loc_ker[2]]
          int_gs = np.sum(
              self.calcium_intensity_kernel_selectors * intens_g[None, :],
              axis=1
          ) / self.cikernelnorm  # shape: (5,)

          # Ratio of green to red (avoid division by zero)
          ci_int_sing = int_gs / (int_rs + 1e-8)

          # Save the mean ratio in [0], the std dev of ratio in [1]
          ci_val = ci_int_sing[0]  # or maybe np.mean(ci_int_sing)
          # If you want the std across the 5 subkernels:
          self.ci_int[neuron_idx][frame_i][1] = np.nanstd(ci_int_sing)

      else:
          # Single-channel workflow (assume red only). 
          # The original code does int_rs / 255, but you can change that as needed.
          ci_int_sing = int_rs / 255.0
          ci_val = ci_int_sing[0]  # or np.mean(ci_int_sing)
          std = np.nan  # no std for single-channel, or use np.nanstd if you want
      self.ci_int[neuron_idx][frame_i][0] = ci_val
      self.ci_int[neuron_idx][frame_i][1] = std
      return [ci_val, std]
    def update_ci(self, dataset, t=None, i_from1=None):
        if dataset.point_data:
            self._update_ci_from_pointdat(dataset, t, i_from1)
        else:
            raise NotImplementedError("CalciumAnalyzer.update_ci() is not implemented for non-point data datasets.")


    
    def change_num_neurons(self, num_neurons):
        current_num_neurons = self.ci_int.shape[0]
        current_ci = self.ci_int
        if current_num_neurons < num_neurons:
            self.ci_int = np.full((num_neurons, self.num_frames, 2), np.nan, dtype=np.float32)
            self.ci_int[:current_num_neurons] = current_ci
        else:
            self.ci_int = self.ci_int[:num_neurons]
