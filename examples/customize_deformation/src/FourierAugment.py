import torch
import torch.nn as nn
import torch.fft
import numpy as np

class Fourier(nn.Module):
    def __init__(self,realshape,k_cut_dimless=4):
        super(Fourier,self).__init__()
        assert all([d%2==0 for d in realshape]),"Only Even dimensions allowed"
        self.realshape=realshape
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
        ks=[np.fft.fftfreq(d,1/d) for d in self.shape[:-1]]
        ks.append(np.arange(self.shape[-1]))
        self.kgrid=np.array(np.meshgrid(*ks,indexing="ij"))
        self.ksqabs=np.sum(self.kgrid**2,axis=0)
        arrs=[torch.arange(d) for d in self.shape]
        unique_holders=torch.stack(torch.meshgrid(*arrs))
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
        self.dim=torch.sum(feed_ind).item()

    def forward(self,x):
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
        return torch.fft.irfftn(self.f_k,s=self.realshape)
    
    def to(self,**kwargs):
        self.feed_ind=self.feed_ind.to(**kwargs)
    
class FourierDeformationPk(nn.Module):
    def __init__(self,realshape,k_cut_dimless=4,Pk=lambda k:1/(k**2+1),dimscale=(1,1,1),defscale=(1,1,1)):
        super(FourierDeformationPk,self).__init__()
        self.realshape=realshape
        self.vecdim=len(self.realshape)
        assert self.vecdim in [2,3],"dim only 2 and 3 allowed"
        self.f=Fourier(realshape,k_cut_dimless=k_cut_dimless)
        self.size=np.prod(realshape)
        self.dimscale=np.array(dimscale)
        self.ks=torch.tensor(np.sum((self.f.kgrid*self.dimscale[:,None,None,None])**2,axis=0)).repeat(self.f.feed_ind.size(0),*[1 for _ in range(self.vecdim)])[self.f.feed_ind]
        self.defscale=torch.tensor(np.array(defscale)/self.dimscale).to(dtype=torch.float32)
        self.Pk=Pk(self.ks).clone().detach().to(dtype=torch.float32)*self.size
        self.dim=self.f.dim*self.vecdim
        self.grid=torch.stack(torch.meshgrid(*[torch.arange(self.realshape[i]) for i in range(self.vecdim)]))
        self.grid.requires_grad=False
        if self.vecdim==2:
            self.normten=torch.tensor(self.realshape)[None,:,None,None]-1
        elif self.vecdim==3:
            self.normten=torch.tensor(self.realshape)[None,:,None,None,None]-1
            
    def forward(self,fr,mask=None,individual_seeds=False,single_batch=False):
        if individual_seeds:
            assert False, "not implemented"
        if single_batch:
            fr=fr.unsqueeze(0)
            if mask is not None:
                mask=mask.unsqueeze(0)
        seeds=torch.randn(1,self.dim).to(fr.device,dtype=torch.float32)
        self.disp=self.f(seeds.reshape(seeds.size(0),self.vecdim,self.f.dim)*self.Pk[None,None,:]*self.defscale[None,:,None])
        moved=(2*((self.grid[None]-self.disp)/self.normten)-1).to(dtype=torch.float32,device=seeds.device)
        if self.vecdim==2:
            moved=moved.permute(0,2,3,1)[...,[1,0]]
        elif self.vecdim==3:
            moved=moved.permute(0,2,3,4,1)[...,[2,1,0]]
        fr_aug=torch.nn.functional.grid_sample(fr.to(torch.float32),moved.repeat(fr.size(0),*[1 for _ in range(self.vecdim+1)]), mode='bilinear', padding_mode="border",align_corners=True)
        if single_batch:
            fr_aug=fr_aug[0]
        if mask is not None:
            mask_aug=torch.nn.functional.grid_sample(mask.unsqueeze(1).to(torch.float32),moved.repeat(mask.size(0),*[1 for _ in range(self.vecdim+1)]), mode='nearest', padding_mode='zeros',align_corners=True)[:,0].to(torch.long)
            if single_batch:
                mask_aug=mask_aug[0]
            return fr_aug,mask_aug
        return fr_aug
    
    def to(self,**kwargs):
        self.defscale=self.defscale.to(**kwargs)
        self.f.to(**kwargs)
        self.Pk=self.Pk.to(**kwargs)
        self.grid=self.grid.to(**kwargs)
        self.normten=self.normten.to(**kwargs)
        
        
