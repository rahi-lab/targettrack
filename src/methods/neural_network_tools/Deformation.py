import torch
import torch.nn as nn
import torch.fft
import numpy as np

class FourierLowK(nn.Module):
    def __init__(self,realshape,k_cut_dimless=4):
        super().__init__()
        assert all([d%2==0 for d in realshape]),"Only Even dimensions allowed"
        self.realshape=realshape
        self.scale=np.prod(realshape)
        shape=list(realshape)
        shape[-1]=shape[-1]//2+1
        self.shape=tuple(shape)
        self.n_d=len(shape)
        mids=np.array(shape)//2
        mids[-1]=shape[-1]-1
        self.mids=tuple(mids)
        self.init_feed_ind(k_cut_dimless)

    def hermitian_symmetric(self,f_k_half):
        if self.n_d==2:
            f_k_half[self.mids[0]+1:,0]=torch.flip(f_k_half[1:self.mids[0],0],dims=(0,)).conj()
            f_k_half[self.mids[0]+1:,self.mids[1]]=torch.flip(f_k_half[1:self.mids[0],self.mids[1]],dims=(0,)).conj()
        elif self.n_d==3:
            f_k_half[self.mids[0]+1:,0,0]=torch.flip(f_k_half[1:self.mids[0],0,0],dims=(0,)).conj().clone()
            f_k_half[self.mids[0]+1:,0,self.mids[2]]=torch.flip(f_k_half[1:self.mids[0],0,self.mids[2]],dims=(0,)).conj().clone()
            f_k_half[self.mids[0]+1:,self.mids[1],0]=torch.flip(f_k_half[1:self.mids[0],self.mids[1],0],dims=(0,)).conj().clone()
            f_k_half[self.mids[0]+1:,self.mids[1],self.mids[2]]=torch.flip(f_k_half[1:self.mids[0],self.mids[1],self.mids[2]],dims=(0,)).conj().clone()

            f_k_half[0,self.mids[1]+1:,0]=torch.flip(f_k_half[0,1:self.mids[1],0],dims=(0,)).conj().clone()
            f_k_half[0,self.mids[1]+1:,self.mids[2]]=torch.flip(f_k_half[0,1:self.mids[1],self.mids[2]],dims=(0,)).conj().clone()
            f_k_half[self.mids[0],self.mids[1]+1:,0]=torch.flip(f_k_half[self.mids[0],1:self.mids[1],0],dims=(0,)).conj().clone()
            f_k_half[self.mids[0],self.mids[1]+1:,self.mids[2]]=torch.flip(f_k_half[self.mids[0],1:self.mids[1],self.mids[2]],dims=(0,)).conj().clone()

            f_k_half[self.mids[0]+1:,self.mids[1]+1:,0]=torch.flip(f_k_half[1:self.mids[0],1:self.mids[1],0],dims=(0,1)).conj().clone()
            f_k_half[self.mids[0]+1:,1:self.mids[1]:,0]=torch.flip(f_k_half[1:self.mids[0],self.mids[1]+1:,0],dims=(0,1)).conj().clone()
            f_k_half[self.mids[0]+1:,self.mids[1]+1:,self.mids[2]]=torch.flip(f_k_half[1:self.mids[0],1:self.mids[1],self.mids[2]],dims=(0,1)).conj().clone()
            f_k_half[self.mids[0]+1:,1:self.mids[1]:,self.mids[2]]=torch.flip(f_k_half[1:self.mids[0],self.mids[1]+1:,self.mids[2]],dims=(0,1)).conj().clone()
        else:
            assert False, "n_d=2 or 3"
        return f_k_half

    def init_feed_ind(self,k_cut_dimless):
        ks=[np.fft.fftfreq(d,1/d) for d in self.shape[:-1]]#descrete fourier transform sample frequencies
        ks.append(np.arange(self.shape[-1]))
        self.kgrid=np.array(np.meshgrid(*ks,indexing="ij"))#grids of the fourier space
        self.ksqabs=np.sum(self.kgrid**2,axis=0)
        arrs=[torch.arange(d) for d in self.shape]
        unique_holders=torch.stack(torch.meshgrid(*arrs))#grids of the image
        holders_hs=torch.stack([self.hermitian_symmetric(unique_holder.clone()) for unique_holder in unique_holders])
        same=torch.all(holders_hs==unique_holders,dim=0)
        rem=same*(self.ksqabs<=k_cut_dimless**2)
        feed_ind=torch.stack([rem,rem])
        if self.n_d==2:
            feed_ind[1,0,0]=0
            feed_ind[1,self.mids[0],0]=0
            feed_ind[1,0,self.mids[1]]=0
            feed_ind[1,self.mids[0],self.mids[1]]=0
        elif self.n_d==3:
            feed_ind[1,      0,      0,      0]=0
            feed_ind[1,      0,      0,self.mids[2]]=0
            feed_ind[1,      0,self.mids[1],      0]=0
            feed_ind[1,      0,self.mids[1],self.mids[2]]=0
            feed_ind[1,self.mids[0],      0,      0]=0
            feed_ind[1,self.mids[0],      0,self.mids[2]]=0
            feed_ind[1,self.mids[0],self.mids[1],      0]=0
            feed_ind[1,self.mids[0],self.mids[1],self.mids[2]]=0
        else:
            assert False, "n_d=2 or 3"
        self.feed_ind=feed_ind.to(dtype=torch.bool)
        self.feed_ind.requires_grad=False
        self.dim=torch.sum(feed_ind).item()#converts the sum of all elements into python scaler

    def forward(self,x):
        x=x*self.scale
        b,c=x.size(0),x.size(1)
        x=x.reshape(b*c,-1)
        arr=torch.zeros((x.size(0),2,*self.shape),dtype=torch.float32,device=x.device)
        arrcomps=torch.zeros((x.size(0),*self.shape),dtype=torch.complex64,device=x.device)
        arr[:,self.feed_ind]=x
        arrcomps.real+=arr[:,0]
        arrcomps.imag+=arr[:,1]
        outs=[]
        for arrcomp in arrcomps:
            outs.append(self.hermitian_symmetric(arrcomp.clone())[None])
        x=torch.cat(outs,dim=0)
        s=list(x.size())
        self.f_k=x.reshape(b,c,*s[1:])
        return torch.fft.irfftn(self.f_k,s=self.realshape)#computes n dimensional inverse descrete fourier transform of the real input- imaginary components ignored

    def to(self,**kwargs):
        self.feed_ind=self.feed_ind.to(**kwargs)

def get_deformation(ptfrom,ptto,sh,k_cut_dimless=2.5,lr=0.1,iterations=200,lambda_div=1,scale=(1,1,1),at_least=5,device="cpu",verbose=False,return_losses=False):
    vecs=(ptto-ptfrom)
    valids=np.nonzero(np.all(np.isnan(vecs)==0,axis=1))[0]
    if len(valids)<at_least:
        return None,"Not enough points"
    vecs=vecs[valids][:,:]
    locs=ptfrom[valids][:,:]
    W,H,D=sh

    f=FourierLowK((W,H,D),k_cut_dimless=k_cut_dimless)

    locs_gridded=2*(torch.tensor(locs)[:,None,None,None,:]/(np.array([W,H,D])[None,None,None,None,:]-1))-1#MB: I rhink with None we add unit axis
    locs_gridded=locs_gridded[...,[2,1,0]].to(device=device,dtype=torch.float32)
    vecs_target=torch.tensor(vecs).to(device=device,dtype=torch.float32)

    x=torch.zeros(1,3,f.dim)
    #initialize the displacement field as the mean displacement
    x[0,:,0]=torch.tensor(np.mean(vecs,axis=0))
    x=x.to(device=device)
    x=torch.nn.Parameter(x)
    opt=torch.optim.Adam([x],lr=lr)
    if return_losses:
        losses=[]
    for iters in range(iterations):
        deformation=f(x)
        #deformation=torch.mean(deformation,dim=4,keepdim=True).repeat(1,1,1,1,deformation.size(4))#mean over z axis
        vecs_sampled=torch.nn.functional.grid_sample(deformation.repeat(locs_gridded.size(0),1,1,1,1),locs_gridded, mode='bilinear', padding_mode='border',align_corners=True)[:,:,0,0,0]
        loss=torch.nn.functional.l1_loss(vecs_sampled,vecs_target)#this is where the points that are mapped to each other are included
        gx=deformation[:,0,2:,1:-1,1:-1]-deformation[:,0,:-2,1:-1,1:-1]#how far is each pixel from the previous one in the direction of x
        gy=deformation[:,1,1:-1,2:,1:-1]-deformation[:,1,1:-1,:-2,1:-1]
        gz=deformation[:,2,1:-1,1:-1,2:]-deformation[:,2,1:-1,1:-1,:-2]
        divergence=scale[0]*gx+scale[1]*gy+scale[2]*gz
        loss+=lambda_div*torch.mean(torch.abs(divergence))
        #print(f.kgrid.shape)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if return_losses:
            losses.append(loss.item())
        if verbose:
            print(loss.item())
    if return_losses:
        return deformation,losses,"success"
    return deformation,"success"

def deform(sh,deformation,frame,mask=None):
    W,H,D=sh
    grid=torch.stack(torch.meshgrid(*[torch.arange(s) for s in (W,H,D)]),dim=0).to(dtype=torch.float32,device=deformation.device)
    normten=(torch.tensor([W,H,D])[None,:,None,None,None]-1).to(dtype=torch.float32,device=deformation.device)
    moved=(2*((grid[None]-deformation)/normten)-1).to(dtype=torch.float32,device=deformation.device)
    moved=moved.permute(0,2,3,4,1)[...,[2,1,0]]
    fr_aug=torch.nn.functional.grid_sample(frame.to(torch.float32),moved.repeat(frame.size(0),1,1,1,1), mode='bilinear', padding_mode="border",align_corners=True)
    if mask is not None:
        mask_aug=torch.nn.functional.grid_sample(mask.unsqueeze(1).to(torch.float32),moved.repeat(mask.size(0),1,1,1,1), mode='nearest', padding_mode='zeros',align_corners=True)[:,0].to(torch.long)
        return fr_aug,mask_aug
    return fr_aug
