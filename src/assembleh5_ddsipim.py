#These are instructions to assemble a minimal h5 file compatible with our GUI
import h5py
import numpy as np
import sys
import os

if len(sys.argv)>1:
    fn=sys.argv[1]
else:
    fn="./data/ddsipim_data_1.h5" # created targettrack dataset name 

h5=h5py.File(fn,"w")

C=2#number of channels
W=2292#x -> most image processing sometimes use y first (H,W,D),but we use x first
H=1180# y
D=10#z
T=300# time-> this is for the main volume
N_neurons=5#in case we already know how many neurons we have
h5_dir = "" # path to the extracted .ome.tif dataset to .h5


for i in range(T):
    filename = os.path.join(h5_dir, f"timepoint_{i}.h5")
    with h5py.File(filename, 'r') as file:
        vol1 = file[f'{0}'][:,:,:]

    volumes = np.expand_dims(vol1, axis=0)
    transposed_volumes = volumes.transpose(0, 3, 2, 1)
    dset=h5.create_dataset(str(i)+"/frame",(C,W,H,D),dtype="i2",compression="gzip")
    dset[...]=transposed_volumes.astype(np.int16)
    print('Timestamp: ',i)

#these are meta data, attributes
h5.attrs["name"]="examplefile"
h5.attrs["C"]=C
h5.attrs["W"]=W
h5.attrs["H"]=H
h5.attrs["D"]=D
h5.attrs["T"]=T
h5.attrs["N_neurons"]=N_neurons
h5.close()
