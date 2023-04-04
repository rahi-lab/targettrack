import torch
import numpy as np
import scipy.spatial as sspat
import albumentations as A
from torch.utils.data import Dataset
import os
import glob
import cc3d
import scipy.stats as sstats

def get_maskpts(labels,coords,gridpts,radius=4):
    tree=sspat.cKDTree(coords)
    ds,iis=tree.query(gridpts,k=1)
    maskpts=labels[iis]
    return maskpts*(ds<=radius)

def get_point_fast3D(maskflat,gridpts,N_labels):
    return sstats.binned_statistic(maskflat,gridpts.T,bins=np.arange(0,N_labels+1)-0.5,statistic="mean")[0].T

def get_pts_dict(mask,grid,weight=None):
    dim=len(mask.shape)
    assert dim==2 or dim==3
    labels_out,maxlab=cc3d.connected_components(mask, connectivity=4 if dim==2 else 6, return_N=True)

    labels=[]
    sizes=[]
    cache_dict=[]
    for i in range(1,maxlab+1):
        one=(labels_out==i)
        cache_dict.append(one)
        labels.append(mask[one][0])
        sizes.append(np.sum(one))
    args=np.argsort(sizes)[::-1]

    pts_dict={0:None}
    if weight is None:
        weight=np.ones(grid.shape[1:])
    wgrid=grid*weight[None,...]
    for i in args:#decending
        l=labels[i]
        if l in pts_dict.keys():
            continue
        wvec=np.sum(wgrid[:,cache_dict[i]],axis=1)
        norm=np.sum(weight[cache_dict[i]])
        if norm==0:
            continue
        pts_dict[l]=wvec/norm
    del pts_dict[0]
    return pts_dict

def selective_ce(pred_raw,target_mask):
    existing=torch.unique(target_mask)
    with torch.no_grad():
        trf=torch.zeros(torch.max(existing)+1).to(device=pred_raw.device,dtype=torch.long)
        trf[existing]=torch.arange(0,len(existing)).to(device=pred_raw.device,dtype=torch.long)
        mask=trf[target_mask]
        mask.requires_grad=False
    return torch.nn.functional.cross_entropy(pred_raw[:,existing],mask)

def get_additional_inds(traininds,distmat):
    T=distmat.shape[0]
    traininds_aug=np.array(traininds).copy()
    while len(traininds_aug)<T:
        not_tr=np.ones(T,dtype=bool)
        not_tr[traininds_aug]=0
        not_traininds=np.nonzero(not_tr)[0]#MB: index of frames that are not in training set
        subdist=distmat[traininds_aug]#MB: distance matrix for frames
        subdist=subdist[:,not_tr]
        dists=np.min(subdist,axis=0)#min of each colums, minimum distance of each not-trained frame from the training set
        new=not_traininds[np.argmax(dists)]#index of a frameoutside of training set that has the maximum dist
        traininds_aug=np.append(traininds_aug,new)#MB:add the frame corresponding to index 'new' to the training set
    return traininds_aug[len(traininds):]
    
def single_batch(batch,network,lossfunc,optimizer,device):
    loss=lossfunc(network(batch[0].to(device=device,dtype=torch.float32)),batch[1].to(device=device,dtype=torch.long))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

class SingleData():
    def __init__(self,frame,mask,data_type="gt"):
        self.frame=frame
        self.mask=mask
        self.data_type=data_type

    def augment(self,trf_func):
        return SingleData(*trf_func(self),"augmented")

class TrainDataset(Dataset):
    def __init__(self,shape):
        super().__init__()
        self.shape=shape
        self.datas=[]

        inf={}
        inf["mask"]="mask"
        self.aff=A.Compose([A.ShiftScaleRotate(shift_limit=0.12,scale_limit=0.12,rotate_limit=30,interpolation=1,border_mode=0,value=0,mask_value=0,p=1),
            ],
            additional_targets=inf
            )
        self.comp=A.Compose([A.ShiftScaleRotate(shift_limit=0.5,scale_limit=0.5,rotate_limit=180,interpolation=1,border_mode=0,value=0,mask_value=0,p=1),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.OpticalDistortion(distort_limit=0.3,shift_limit=0.3,border_mode=0,p=0.7),
            A.MotionBlur(blur_limit=13,p=0.7),
            ],
            additional_targets=inf
            )
        self.grid2d=None

    def add_data(self,frame,mask,data_type):
        frame=(frame/255).to(dtype=torch.float32)
        self.datas.append(SingleData(frame,mask,data_type))
        self.update_length()
    
    def update_length(self):
        self.num_frames_tot=len(self.datas)

    def cut(self,feed,p=1):
        if np.random.random()<p:
            point=np.random.random(2)*np.array(self.shape[1:3])
            m=np.tan(np.pi*(np.random.random()-0.5))
            if np.isnan(m) or np.isinf(m):
                m=1
            top_or_bottom=np.random.random()>0.5
            if self.grid2d is None:
                self.grid2d=np.array(np.meshgrid(np.arange(self.shape[1]),np.arange(self.shape[2]),indexing="ij"))
            valid=(self.grid2d[1]-point[1])>(m*(self.grid2d[0]-point[0]))
            if valid.sum()<((self.shape[1]*self.shape[2])/2):
                valid=~valid
            feed["image"]*=valid[:,:,None]
            feed["mask"]*=valid[:,:,None]
        return feed

    def get_trf(self,data,mode="comp_cut"):
        if mode=="none":
            return fr,mask
        fr,mask=data.frame,data.mask
        fr=fr.numpy()
        mask=mask.numpy()
        feed={}
        feed["mask"]=mask #z is automatically channel

        if mode=="comp_cut":
            fr=self.CD_to_end(fr)
            if data.data_type=="ta":
                res=self.aff(image=fr,**feed)
            else:
                res=self.comp(image=fr,**feed)
            res=self.cut(res,p=0.5)
            fr=self.CD_back(res["image"])
            mask=res["mask"]
            return torch.tensor(fr),torch.tensor(mask)

        elif mode=="comp":
            fr=self.CD_to_end(fr)
            if data.data_type=="ta":
                res=self.aff(image=fr,**feed)
            else:
                res=self.comp(image=fr,**feed)
            fr=self.CD_back(res["image"])
            mask=res["mask"]
            return torch.tensor(fr),torch.tensor(mask)
        else:
            assert False,"wrong trf(augmentation) mode"

    def CD_to_end(self,fr):
        return fr.transpose(1,2,0,3).reshape(self.shape[1],self.shape[2],-1)

    def CD_back(self,fr):
        return fr.reshape(self.shape[1],self.shape[2],self.shape[0],-1).transpose(2,0,1,3)

    def __getitem__(self,i):
        assert 0<=i<self.num_frames_tot
        data=self.datas[i]
        data=data.augment(self.get_trf)
        return data.frame,data.mask

    def __len__(self):
        return self.num_frames_tot

class EvalDataset(Dataset):
    def __init__(self,folpath,T,mask=False,maxz=False):
        super().__init__()
        self.maxz=maxz
        self.folpath=folpath
        self.framefolpath=os.path.join(self.folpath,"frames")
        self.mask=mask
        if self.mask:
            self.maskfolpath=os.path.join(self.folpath,"masks")
        self.num_frames_tot=T


    def __getitem__(self,i):
        assert 0<=i<self.num_frames_tot
        fr=(torch.load(os.path.join(self.framefolpath,str(i)+".pt"))/255).to(torch.float32)
        if self.mask:
            maskfp=os.path.join(self.maskfolpath,str(i)+".pt")
            if (not self.maxz) and os.path.exists(maskfp):
                mask=torch.load(maskfp).to(torch.long)
            else:
                mask=None
            if self.maxz:
                return fr.max(3)[0],None
            else:
                return fr,mask
        if self.maxz:
            return fr.max(3)[0]
        else:
            return fr

    def __len__(self):
        return self.num_frames_tot


