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
