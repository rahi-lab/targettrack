import sys
import os
import h5py
import scipy.ndimage as sim
import numpy as np
import torch
import NNtools
import shutil
import multiprocessing
import sklearn.metrics as skmet
import cc3d


#### TODO: use argparse ####
assert len(sys.argv)==4, "1st argument: dataset_h5_path, 2nd argument: neural_network_name, 3rd argument: runname"

verbose=True

dataset_path=sys.argv[1]
NetName=sys.argv[2]
runname=sys.argv[3]
identifier="net/"+NetName+"_"+runname

thres=4

chunksize=30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#### Open the dataset ####
dataset_name=os.path.split(dataset_path)[1]
h5=h5py.File(dataset_path,"r+")

#### What we need from the dataset ####
dependencies=["W","H","D","T","N_neurons"]
for dep in dependencies:
    if dep not in h5.attrs.keys():
        h5.close()
        assert False, "Dependency "+dep+" not in  attributes"
if identifier not in h5.keys():
    h5.close()
    assert False, identifier+" not in  dataset"
def gen_tracked_points(frame,pred,thres):
    cutpred=(frame>thres)*pred
    res=cc3d.connected_components(cutpred, connectivity=6)
    labels=[]
    sizes=[]
    cache_dict={}
    for i in range(np.max(res)+1):
        one=(res==i)
        cache_dict[i]=one
        labels.append(cutpred[one][0])
        sizes.append(np.sum(one))
    args=np.argsort(sizes)[::-1]
    pts_dict={0:None}

    wgrid=grid*frame[...,None]
    for i in args:#decending
        l=labels[i]
        if l in pts_dict.keys():
            continue
        wvec=np.sum(wgrid[cache_dict[i]],axis=0)
        if np.sum(wvec)==0:
            continue
        pts_dict[l]=wvec/np.sum(frame[cache_dict[i]])

    pts=np.full((num_classes,3),np.nan)
    for key,val in pts_dict.items():
        if key!=0:
            pts[key]=val
    return pts

def get_pts_iou(package):
    fr,predmask,mask=package
    if mask is None:
        iou=np.full((num_classes,num_classes),np.nan)
    else:
        iou=skmet.confusion_matrix(mask.flatten(),predmask.flatten(),labels=labels)
    return [gen_tracked_points(fr,predmask,thres=thres),iou]

def pack(i):
    if verbose:
        print(i)
    fr=np.array(h5[str(i)+"/frame"])[0]
    predmask=np.array(h5[identifier][str(i)+"/predmask"])
    if str(i)+"/mask" in h5.keys():
        mask=np.array(h5[str(i)+"/mask"])
    else:
        mask=None
    return [fr,predmask,mask]
try:
    num_classes=h5.attrs["N_neurons"]+1
    labels=np.arange(num_classes)

    grid=np.moveaxis(np.array(np.meshgrid(np.arange(h5.attrs["W"]),np.arange(h5.attrs["H"]),np.arange(h5.attrs["D"]),indexing="ij")),0,3)

    ptss=np.full((h5.attrs["T"],num_classes,3),np.nan)
    ious=np.full((h5.attrs["T"],num_classes,num_classes),np.nan)


    pool=multiprocessing.Pool(multiprocessing.cpu_count())

    result=pool.imap(get_pts_iou,(pack(i) for i in range(h5.attrs["T"])),chunksize=chunksize)

    for idx,res in enumerate(result):
        pts,iou=res
        ptss[idx]=pts
        ious[idx]=iou
    if "NN_pointdat" not in h5[identifier].keys():
        h5[identifier].create_dataset("NN_pointdat",ptss.shape,dtype="f4")
    h5[identifier]["NN_pointdat"][...]=ptss
    if "ious" not in h5[identifier].keys():
        h5[identifier].create_dataset("ious",ious.shape,dtype="i2")
    h5[identifier]["ious"][...]=ious
    if verbose:
        print("Extraction successful")
except Exception as exception:
    print(exception)
