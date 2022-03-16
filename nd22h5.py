import sys
import os
import numpy as np
import h5py
from nd2reader import ND2Reader


nd2filename=sys.argv[1]
hdfout=sys.argv[2]

with ND2Reader(nd2filename) as images:
    c = images.sizes['c']
    x = images.sizes['x']
    y = images.sizes['y']
    z = images.sizes['z']
    images.iter_axes = 't'
    images.bundle_axes = 'xyz'

    # compute number of frames, which is wrong in the metadata   (by dichotomy)
    max_n_frames = len(images.metadata["frames"])
    mi = 0
    ma = max_n_frames
    prev_t = -1  # anything that is not the starting point
    while True:
        t = mi + (ma - mi) // 2
        try:
            images[t]
            mi = t
        except:
            ma = t
        if prev_t == t:
            break
        prev_t = t
    t += 1

    with h5py.File(hdfout, 'w') as hf:
        for i1 in range(t):
            im3d = np.zeros((c, x, y, z))
            dset = hf.create_dataset(str(i1) + "/frame", (c, x-2, y, z), dtype="i2", compression="gzip")
            if c>1:
                for c1 in range(c):
                    images.default_coords['c'] = c1
                    im3d[1-c1] = np.array(images[i1]).astype(np.int32)
            else:
                im3d[0] = np.array(images[i1]).astype(np.int32)
            dset[...] = im3d[:,:-2,:,:]
        name1 = os.path.basename(hdfout)
        name1 = name1.split(".")
        name = name1[0]
        print(name)
        hf.attrs["name"]=name#os.path.basename(hdfout)
        hf.attrs["C"]=c
        hf.attrs["W"]=y
        hf.attrs["H"]=x-2
        hf.attrs["D"]=z
        hf.attrs["T"]=t
        hf.attrs["N_neurons"]=0
