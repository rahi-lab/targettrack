#These are instructions to assemble a minimal h5 file compatible with our GUI
import h5py
import numpy as np
import sys
if len(sys.argv)>1:
    fn=sys.argv[1]
else:
    fn="./data/example.h5"

h5=h5py.File(fn,"w")

C=2#number of channels
W=256#x -> most image processing sometimes use y first (H,W,D),but we use x first
H=160#y
D=16#z
T=25# time-> this is for the main volume
N_neurons=40#in case we already know how many neurons we have


for i in range(T):
    print(i)#just for check
    dset=h5.create_dataset(str(i)+"/frame",(C,W,H,D),dtype="i2",compression="gzip")
    dset[...]=(np.random.random((C,W,H,D))*255).astype(np.int16)#int16 is i2
    dset=h5.create_dataset(str(i)+"/mask",(W,H,D),dtype="i2",compression="gzip")
    dset[...]=(np.random.randint(0,N_neurons+1,(W,H,D))).astype(np.int16)

#initialize points
dset=h5.create_dataset("pointdat",(T,N_neurons+1,3),dtype="f4")
dset[...]=(np.random.random((T,N_neurons+1,3))*np.array([W,H,D])[None,None,:]).astype(np.float32)
dset[:,0]=np.nan

#these are meta data, attributes
h5.attrs["name"]="examplefile"
h5.attrs["C"]=C
h5.attrs["W"]=W
h5.attrs["H"]=H
h5.attrs["D"]=D
h5.attrs["T"]=T
h5.attrs["N_neurons"]=N_neurons
h5.close()


worm_file = nd2.ND2File(path)
worm_array = nd2.imread(path)
       
def dog_filter(im, sigma1, sigma2):
    """
    Apply a difference of Gaussians filter
    """
    im1 = ndimage.gaussian_filter(im.astype(np.float32), sigma1)
    im2 = ndimage.gaussian_filter(im.astype(np.float32), sigma2)
    return im1 - im2
mean_intensities = np.mean(worm_array_chunk, axis=(2,3))
std_intensities = np.std(worm_array_chunk, axis=(2,3))
'''
From the above, it appears the the microscope is using 40 frames for one volume
starting at index 20. We will use this info to construct our array of 3D frames.
Because the images are taken in alternating order (the microscope images in one 
direction, and then the reverse), we reverse the depth dimension on every other frame


Then,we apply the difference of Gaussians filter to each slice of each frame, 
and save the resulting h5 file as indicated in the targettrack README
'''
fig, axs = plt.subplots(2, 1, figsize = (20, 10), tight_layout = True)
ax = axs[0]
ax.set_title('Intensities')
ax.set_xlabel('Frame')
ax.set_ylabel('Mean Intensity')
ax.plot(mean_intensities[:210,0] - np.mean(mean_intensities[:,0]), label = 'Channel 0', color = 'blue', alpha = 0.5)
ax.plot(mean_intensities[:210,1] - np.mean(mean_intensities[:,1]), label = 'Channel 1', color = 'red', alpha = 0.5)
ax.legend()
axx = axs[1]
axx.set_title('Standard Deviation of Intensities')
axx.set_xlabel('Frame')
axx.set_ylabel('Standard Deviation')
axx.plot(std_intensities[:210,0], label = 'Channel 0', color = 'blue', alpha = 0.5)
axx.plot(std_intensities[:210,1], label = 'Channel 1', color = 'red', alpha = 0.5)
axx.legend()

plt.show()
# CONSTANTS (will need to be changed on experiment by experiment basis until script is written)
import numpy as np
from scipy.signal import find_peaks, savgol_filter

def find_small_valley_minima(data, smoothing_window=11, smoothing_order=3, 
                             distance=20, prominence=0.5, width=5):
    # Smooth the data
    smoothed_data = savgol_filter(data, smoothing_window, smoothing_order)
    
    # Find all minima (negative of maxima)
    minima, _ = find_peaks(-smoothed_data, distance=distance, prominence=prominence, width=width)
    
    return minima



# Assuming your data is stored in y0 and y1 for Channel 0 and Channel 1 respectively
minima_channel0 = find_small_valley_minima(std_intensities[:,0])
minima_channel1 = find_small_valley_minima(std_intensities[:,1])

arr = worm_array_chunk.copy()

START_IDX = 20 #Determined by finding the top/bottom of the volume (which will reliably be min stdev intensity)
FPV = 40 #frames per volume

num_frames = worm_array_chunk.shape[0] - START_IDX
T = num_frames // FPV
D = FPV  #frames per volume is depth
C = worm_file.sizes['C']
H = worm_file.sizes['Y'] #assuming Y coordinate is height
W = worm_file.sizes['X'] #assuming X coordinate is width

print(arr.shape)
extra = num_frames % FPV
if extra > 0:
    arr = arr[START_IDX:-extra,...]
else:
    arr = arr[START_IDX:,...]

arr = arr.reshape(T, D, C, H, W)

print(arr.shape)
arr = arr.transpose(0, 2, 1, 3, 4)
print(arr.shape)
# sigma1 = 1
# sigma2 = 8

# for f in tqdm(range(T)):
#     for d in range(D):
#         if f % 2 == 0:
#             worm_frames[f, 0, d] = dog_filter(arr[f*FPV + d, 0], sigma1, sigma2)
#             worm_frames[f, 1, d] = dog_filter(arr[f*FPV + d, 1], sigma1, sigma2)
#         else:
#             worm_frames[f, 0, D - d - 1] = dog_filter(arr[f*FPV + d, 0], sigma1, sigma2)
#             worm_frames[f, 1, D - d - 1] = dog_filter(arr[f*FPV + d, 1], sigma1, sigma2)








def plot_intensity_vs_frame(arr, start_idx=20, frames_per_volume=40, title = None):
    """
    Plot mean intensities and standard deviations of a 4D array (time, channel, height, width).
    
    Parameters:
    test_array (np.array): 4D array of shape (time, channel, height, width)
    start_idx (int): Starting index of the first volume
    frames_per_volume (int): Number of frames per volume
    """
    mean_intensities = np.mean(arr, axis=(2,3))
    std_intensities = np.std(arr, axis=(2,3))

    fig, axs = plt.subplots(2, 1, figsize=(20, 10), tight_layout=True)

    # Plot mean intensities
    ax = axs[0]
    ax.set_title('Intensities')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Mean Intensity')
    
    xx = np.arange(-start_idx, arr.shape[0] - start_idx)


    ax.plot(xx, mean_intensities[:,0] - np.mean(mean_intensities[:,0]), label='Channel 0', color='blue', alpha=0.5)
    ax.plot(xx, mean_intensities[:,1] - np.mean(mean_intensities[:,1]), label='Channel 1', color='red', alpha=0.5)
    ax.legend()

    # Plot standard deviations
    axx = axs[1]
    axx.set_title('Standard Deviation of Intensities')
    axx.set_xlabel('Frame')
    axx.set_ylabel('Standard Deviation')
    axx.plot(xx, std_intensities[:,0], label='Channel 0', color='blue', alpha=0.5)
    axx.plot(xx, std_intensities[:,1], label='Channel 1', color='red', alpha=0.5)
    axx.legend()

    # Add colored rectangles to show alternating periods
    colors = ['lightgreen', 'lightyellow']
    for i in range(0, test_array.shape[0]-start_idx, frames_per_volume*2):
        for j, ax in enumerate(axs):
            ymin, ymax = ax.get_ylim()
            height = ymax - ymin
            # First period
            rect = Rectangle((i, ymin), frames_per_volume, height, facecolor=colors[0], alpha=0.3)
            ax.add_patch(rect)
            # Second period (reversed)
            rect = Rectangle((i+frames_per_volume, ymin), frames_per_volume, height, facecolor=colors[1], alpha=0.3)
            ax.add_patch(rect)
        
        # Add text to indicate direction
        axs[0].text(i+frames_per_volume/2, axs[0].get_ylim()[1], 'Forward', ha='center', va='top')
        axs[0].text(i+frames_per_volume*1.5, axs[0].get_ylim()[1], 'Reverse', ha='center', va='top')

    # Add vertical lines for volume boundaries
    for i in range(0, test_array.shape[0]-start_idx, frames_per_volume):
        for ax in axs:
            ax.axvline(x=i, color='gray', linestyle='--', alpha=0.5)

    if title:
        fig.suptitle(title, fontsize = 28)
    plt.show()


# Usage example:
# test_array should be a 4D numpy array of shape (time, channel, height, width)
plot_intensity_vs_frame(test_array)



array = worm_array[:, :, Y_MIN:Y_MAX, X_MIN:X_MAX]

# COMMENT THIS OUT TO AVOID PROCESSING ENTIRE ARRAY IF STILL TESTING
# array = test_array 

START_IDX = 20
FPV = 40

D = FPV
num_frames = test_array.shape[0] - START_IDX
T = num_frames // FPV
C = 2
H = array.shape[2]
W = array.shape[3]

extra_frames = num_frames % FPV
if extra_frames > 0:
    tmp_arr = array[START_IDX:-extra_frames]
else:
    tmp_arr = array[START_IDX:]

# Reshape and transpose the array  
tmp_arr = tmp_arr.reshape(T, D, C, H, W).transpose(0, 2, 1, 3, 4)
#Finally align by flipping every other frame along the depth dimension
tmp_arr[1::2] = np.flip(aligned_array[1::2], axis=2)
aligned_array = tmp_arr

print(f"    Aligned Array Shape: {aligned_array.shape} (T x 2 x D x H x W)")

##To double check, we can reverse the reshaping and transposing, and plot
##the full set of frames as we did above
## Reshape and transpose the array back

# tmp_arr = aligned_array.transpose(0, 2, 1, 3, 4).reshape(-1, C, H, W)
# plot_intensity_vs_frame(test_array, start_idx=20, frames_per_volume=40, title = 'Original Array')
# plot_intensity_vs_frame(tmp_arr, start_idx=0, frames_per_volume=40, title = 'Aligned Array')



def dog_filter_2d(im, sigma1, sigma2):
    """Apply a difference of Gaussians filter"""
    im1 = gaussian_filter(im.astype(np.float32), sigma1)
    im2 = gaussian_filter(im.astype(np.float32), sigma2)
    return im1 - im2

# Ensure filtered_array is float32 or float64
filtered_array = np.zeros_like(aligned_array, dtype=np.float32)
total_iterations = T * C * D
sig1, sig2=0.5, 4

with tqdm(total=total_iterations, desc="Applying DoG filter") as pbar:
    for t in range(T):
        for c in range(C):
            for d in range(D):
                filtered_array[t, c, d] = dog_filter_2d(aligned_array[t, c, d], sig1, sig2)
                pbar.update(1)

# Normalize and scale the filtered array
filtered_array -= np.min(filtered_array)
filtered_array /= np.max(filtered_array)
filtered_array *= 255

# Convert to int8 after all floating-point operations are done
filtered_array = np.clip(filtered_array, 0, 255).astype(np.int8)


# Define output path and file name
output_path = "/nese/mit/group/boydenlab/Konstantinos/h5_files"
filename = 'adult_1070'

# Ensure the output directory exists
os.makedirs(output_path, exist_ok=True)
N_neurons = 20 #so targettrack doesn't complain


# Save to HDF5 file
with h5py.File(os.path.join(output_path, filename+".h5"), 'w') as h5:
    # Save each time point
    for i in range(T):
        print(f"Saving time point {i}")
        dset = h5.create_dataset(f"{i}/frame", (C, W, H, D), dtype="i2", compression="gzip")
        dset[...] = np.transpose(filtered_array[i], (0, 2, 3, 1))  # Reorder dimensions to (C, W, H, D)
    
    # Set metadata attributes

    h5.attrs["name"] = filename
    h5.attrs["C"] = C
    h5.attrs["W"] = W
    h5.attrs["H"] = H
    h5.attrs["D"] = D
    h5.attrs["T"] = T
    h5.attrs["N_neurons"] = N_neurons

print("Finished saving the filtered array to HDF5 file.")
