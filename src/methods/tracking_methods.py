import os
import sys
sys.path.append(os.getcwd())
currmod=sys.modules[__name__]
from src.methods.DatasetForMethods import *

import os
import shutil
import numpy as np
import scipy.ndimage as sim
import threading
import time
import scipy.spatial as sspat

import matplotlib.pyplot as plt

class NN():
    default_params={"min_points":1,"channels":None,"mask_radius":4,"2D":False,
    "lr":0.01,
    "n_steps":3000,"batch_size":3,"augment":{0:"comp_cut",500:"aff_cut",1000:"aff"},
    "weight_channel":None,

    "Targeted":False,
    "recalcdistmat":True,"n_steps_posture":3000,"batch_size_posture":16,"umap_dim":None,
    "deformparams":{"k_cut_dimless":2.5,"lr":0.1,"iterations":200,"lambda_div":1,"at_least":5},"num_additional":80,
    "pixel_scale":None,

    "DeformAug":False,
    "deformaugparams":{"epsilon":1e-10}
    }
    help=str(default_params)
    def __init__(self,params):
        self.state=""
        self.cancel=False

        params_dict={}
        try:
            for txt in params.strip().split(";"):
                if txt=="":
                    continue
                key,val=txt.split("=")
                params_dict[key]=eval(val)
        except:
            print("Parameter Parse Failure")
        self.params=self.default_params
        self.params.update(params_dict)

        if self.params["2D"]:
            assert not self.params["Targeted"], "Targeted Augementation only in 3D"
        assert self.params["min_points"]>0

    def run(self,file_path):
        from src.methods.neural_network_tools import NNtools
        from src.methods.neural_network_tools import Deformation
        import torch
        from src.methods.neural_network_tools import Networks
        if self.params["Targeted"] and self.params["umap_dim"] is not None:
            import umap
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state=["Preparing",0]
        self.dataset=Dataset(file_path)
        _,file=os.path.split(file_path)
        folname=file.split(".")[0]
        self.folpath=os.path.join("data","data_temp",folname)
        if os.path.exists(self.folpath):
            shutil.rmtree(self.folpath)
            pass
        os.makedirs(self.folpath)
        self.dataset.open()
        self.data_info=self.dataset.get_data_info()
        os.makedirs(os.path.join(self.folpath,"frames"))
        os.makedirs(os.path.join(self.folpath,"masks"))
        #os.makedirs(os.path.join(self.folpath,"log"))

        if self.params["2D"]:
            assert self.data_info["D"]==1,"Input not 2D"
        if (self.params["pixel_scale"] is None) and ("pixel_scale" in self.data_info.keys()):
            self.params["pixel_scale"]=self.data_info["pixel_scale"]
        else:
            self.params["pixel_scale"]=(1,1,1)

        #make files
        if True:
            self.state=["Making Files",0]
            T,N_points,C,W,H,D=self.data_info["T"],self.data_info["N_points"],self.data_info["C"],self.data_info["W"],self.data_info["H"],self.data_info["D"]
            points=self.dataset.get_points()
            min_points=self.params["min_points"]
            channels=self.params["channels"]
            if channels is None:
                channels=np.arange(C)
            n_channels=len(channels)
            grid=np.stack(np.meshgrid(np.arange(W),np.arange(H),np.arange(D),indexing="ij"),axis=0)
            gridpts=grid.reshape(3,-1).T

            #don't think about non existing points
            existing=np.any(~np.isnan(points[:,:,0]),axis=0)
            existing[0]=1#for background
            labels_to_inds=np.nonzero(existing)[0]
            N_labels=len(labels_to_inds)
            inds_to_labels=np.zeros(N_points+1)
            inds_to_labels[labels_to_inds]=np.arange(N_labels)
            for t in range(T):
                if self.cancel:
                    self.quit()
                    return
                image=self.dataset.get_frame(t)[channels]
                image=torch.tensor(image,dtype=torch.uint8)
                torch.save(image,os.path.join(self.folpath,"frames",str(t)+".pt"))
                pts=points[t]
                valid=~np.isnan(pts[:,0])
                if valid.sum()>=min_points:
                    inds=np.nonzero(valid)[0]
                    maskpts=NNtools.get_mask(inds_to_labels[inds],pts[inds],gridpts,self.params["mask_radius"])
                    mask=torch.tensor(maskpts.reshape(W,H,D),dtype=torch.uint8)
                    torch.save(mask,os.path.join(self.folpath,"masks",str(t)+".pt"))
                self.state[1]=int(100*(t+1)/T)

        #make posture space if TA
        if self.params["Targeted"] and ((not self.dataset.exists("distmat")) or self.params["recalcdistmat"]):
            self.state=["Embedding Posture Space Training",0]
            self.encnet=Networks.AutoEnc2d(sh2d=(W,H),n_channels=n_channels,n_z=min(20,T//4))
            self.encnet.to(device=self.device,dtype=torch.float32)
            self.encnet.train()

            data=NNtools.EvalDataset(folpath=self.folpath,T=T,maxz=True)
            loader=torch.utils.data.DataLoader(data, batch_size=self.params["batch_size_posture"],shuffle=True,num_workers=0,pin_memory=True)
            opt=torch.optim.Adam(self.encnet.parameters(),lr=self.params["lr"])
            n_steps_posture=self.params["n_steps_posture"]

            self.encnet.train()
            traindone=False
            stepcount=0
            #f=open(os.path.join(self.folpath,"log","enc_loss.txt"),"w")
            while not traindone:
                for i,ims in enumerate(loader):
                    if self.cancel:
                        self.quit()
                        return
                    ims=ims.to(device=self.device,dtype=torch.float32)
                    res,latent=self.encnet(ims)
                    loss=torch.nn.functional.mse_loss(res,ims)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    stepcount+=1
                    self.state[1]=int(100*(stepcount/n_steps_posture))
                    #print(stepcount,loss.item())
                    #f.write(str(loss.item())+"\n")
                    if stepcount==n_steps_posture:
                        traindone=True
                        break
            #f.close()

            self.state=["Embedding Posture Space Evaluating",0]
            self.encnet.eval()
            vecs=[]
            with torch.no_grad():
                for i in range(T):
                    #print(i,T)
                    if self.cancel:
                        self.quit()
                        return
                    _,latent=self.encnet(data[i].unsqueeze(0).to(device=self.device,dtype=torch.float32))
                    vecs.append(latent[0].cpu().detach().numpy())
                    self.state[1]=int(100*((i+1)/T) )
            vecs=np.array(vecs).astype(np.float32)
            self.dataset.set_data("latent_vecs",vecs,overwrite=True)
            def standardize(vecs):
                m=np.mean(vecs,axis=0)
                s=np.std(vecs,axis=0)
                return (vecs-m)/(s+1e-10)
            vecs=standardize(vecs)
            if self.params["umap_dim"] is not None:
                u_map=umap.UMAP(n_components=self.params["umap_dim"])
                vecs=u_map.fit_transform(vecs)
            distmat=sspat.distance_matrix(vecs,vecs).astype(np.float32)
            self.dataset.set_data("distmat",distmat,overwrite=True)

        #train network
        if True:
            self.state=["Training Network",0]
            if self.params["2D"]:
                self.net=Networks.TwoDCN(n_channels=n_channels,num_classes=N_labels)
            else:
                self.net=Networks.ThreeDCN(n_channels=n_channels,num_classes=N_labels)
            self.net.to(device=self.device,dtype=torch.float32)

            train_data=NNtools.TrainDataset(folpath=self.folpath,shape=(C,W,H,D))
            loader=torch.utils.data.DataLoader(train_data, batch_size=self.params["batch_size"],shuffle=True, num_workers=0,pin_memory=True)
            opt=torch.optim.Adam(self.net.parameters(),lr=self.params["lr"])
            n_steps=self.params["n_steps"]

            self.net.train()
            traindone=False
            stepcount=0
            #f=open(os.path.join(self.folpath,"log","loss.txt"),"w")
            while not traindone:
                if stepcount in self.params["augment"].keys():
                    train_data.change_augment(self.params["augment"][stepcount])
                for i,(ims,masks) in enumerate(loader):
                    if self.cancel:
                        self.quit()
                        return
                    if self.params["2D"]:
                        ims=ims[:,:,:,:,0]
                        masks=masks[:,:,:,0]
                    ims=ims.to(device=self.device,dtype=torch.float32)
                    masks=masks.to(device=self.device,dtype=torch.long)
                    res=self.net(ims)
                    loss=NNtools.selective_ce(res,masks)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    stepcount+=1
                    self.state[1]=int(100*(stepcount/n_steps))
                    #print(stepcount,loss.item())
                    #f.write(str(stepcount)+str(loss.item())+"\n")
                    if stepcount==n_steps:
                        traindone=True
                        break
            #f.close()

        #Add TA frames and train
        #print("TA start")
        if self.params["Targeted"]:
            #Make TA frames
            if True:
                self.state=["Making TA",0]
                self.ta_path=os.path.join(self.folpath,"TA")
                os.makedirs(self.ta_path)
                os.makedirs(os.path.join(self.ta_path,"frames"))
                os.makedirs(os.path.join(self.ta_path,"masks"))

                traininds=np.array(train_data.indlist)
                for train_ind in traininds:
                    frame_name_from=os.path.join(self.folpath,"frames",str(train_ind)+".pt")
                    frame_name_to=os.path.join(self.ta_path,"frames",str(train_ind)+".pt")
                    shutil.copyfile(frame_name_from,frame_name_to)

                    mask_name_from=os.path.join(self.folpath,"masks",str(train_ind)+".pt")
                    mask_name_to=os.path.join(self.ta_path,"masks",str(train_ind)+".pt")
                    shutil.copyfile(mask_name_from,mask_name_to)

                #print("making augs")
                distmat=self.dataset.get_data("distmat")
                additional_inds=NNtools.get_additional_inds(traininds,distmat)
                data=NNtools.EvalDataset(folpath=self.folpath,T=T,mask=True,maxz=False)

                n_added=0
                count=0
                while True:
                    self.state[1]=int(100*((n_added)/self.params["num_additional"]))
                    if count==len(additional_inds):
                        break
                    i=additional_inds[count]
                    i_parent=traininds[np.argmin(distmat[traininds,i])]
                    #print("child:",i,"parent:",i_parent)
                    count+=1

                    #getting points
                    with torch.no_grad():
                        im=data[i][0].to(device=self.device,dtype=torch.float32)
                        maskpred=torch.argmax(self.net(im.unsqueeze(0))[0],dim=0).cpu().detach().numpy()
                        if self.params["weight_channel"] is None:
                            pts_dict=NNtools.get_pts_dict(maskpred,grid,weight=np.ones(grid.shape[1:]).astype(np.float32))
                        else:
                            pts_dict=NNtools.get_pts_dict(maskpred,grid,weight=im[self.params["weight_channel"]].cpu().detach().numpy())
                    pts_child=np.full((N_points+1,3),np.nan)
                    for label,coord in pts_dict.items():
                        pts_child[labels_to_inds[label]]=coord
                    pts_parent=points[i_parent]

                    #print("Points got",pts_child.shape,pts_parent.shape)

                    deformation,msg=Deformation.get_deformation(ptfrom=pts_parent,ptto=pts_child,sh=(W,H,D),scale=self.params["pixel_scale"],device=self.device,**self.params["deformparams"])
                    if msg!="success":
                        #print("Deformation failed")
                        continue
                    #print("Deformation got")
                    im,mask=data[i_parent]
                    im=im.unsqueeze(0).to(device=self.device,dtype=torch.float32)
                    mask=mask.unsqueeze(0).to(device=self.device,dtype=torch.long)
                    im,mask=Deformation.deform((W,H,D),deformation,im,mask=mask)
                    im=torch.clip(im[0]*255,0,255).to(device="cpu",dtype=torch.uint8)
                    mask=mask[0].to(device="cpu",dtype=torch.uint8)

                    ind=T+i
                    torch.save(im,os.path.join(self.ta_path,"frames",str(ind)+".pt"))
                    torch.save(mask,os.path.join(self.ta_path,"masks",str(ind)+".pt"))
                    n_added+=1
                    if n_added==self.params["num_additional"]:
                        break

            #print("training")
            #Train with TA frames
            if True:
                self.state=["Training TA",0]
                train_data=NNtools.TrainDataset(folpath=self.ta_path,shape=(C,W,H,D))
                train_data.change_augment("aff")
                loader=torch.utils.data.DataLoader(train_data, batch_size=self.params["batch_size"],shuffle=True, num_workers=0,pin_memory=True)
                opt=torch.optim.Adam(self.net.parameters(),lr=self.params["lr"])

                self.net.train()
                traindone=False
                stepcount=0
                while not traindone:
                    for i,(ims,masks) in enumerate(loader):
                        if self.cancel:
                            self.quit()
                            return
                        if self.params["2D"]:
                            ims=ims[:,:,:,:,0]
                            masks=masks[:,:,:,0]
                        ims=ims.to(device=self.device,dtype=torch.float32)
                        masks=masks.to(device=self.device,dtype=torch.long)
                        res=self.net(ims)
                        loss=NNtools.selective_ce(res,masks)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                    stepcount+=1
                    self.state[1]=int(100*(stepcount/n_steps))
                    if stepcount==n_steps:
                        traindone=True
                        break

        #extract points and save
        if True:
            self.state=["Extracting Points",0]
            ptss=np.full((T,N_points+1,3),np.nan)
            if self.params["2D"]:
                grid=grid[:2,:,:,0]
                if self.params["weight_channel"] is None:
                    weight=np.ones(grid.shape[1:]).astype(np.float32)
            else:
                if self.params["weight_channel"] is None:
                    weight=np.ones(grid.shape[1:]).astype(np.float32)
            data=NNtools.EvalDataset(folpath=self.folpath,T=T,maxz=False)

            self.net.eval()
            with torch.no_grad():
                for i in range(T):
                    if self.cancel:
                        self.quit()
                        return
                    im=data[i].to(device=self.device,dtype=torch.float32)
                    if self.params["2D"]:
                        im=im[:,:,:,0]
                    maskpred=torch.argmax(self.net(im.unsqueeze(0))[0],dim=0).cpu().detach().numpy()

                    if self.params["weight_channel"] is None:
                        pts_dict=NNtools.get_pts_dict(maskpred,grid,weight=weight)
                    else:
                        im=im.cpu().detach().numpy()
                        pts_dict=NNtools.get_pts_dict(maskpred,grid,weight=im[self.params["weight_channel"]])
                    if self.params["2D"]:
                        for label,coord in pts_dict.items():
                            ptss[i,label,:2]=coord
                            ptss[i,label,2]=0
                    else:
                        for label,coord in pts_dict.items():
                            ptss[i,label]=coord
                    self.state[1]=int(100*((i+1)/T))
            ptss[:,0,:]=np.nan

            self.dataset.set_data("helper_NN",ptss,overwrite=True)

        self.dataset.close()
        #shutil.rmtree(self.folpath)
        self.state="Done"

    def quit(self):
        shutil.rmtree(self.folpath)

class NPS():
    default_params={"anchor_labels":None,"anchor_times":None,"radius":50}
    help=str(default_params)
    def __init__(self,params):
        self.state=""
        self.cancel=False

        params_dict={}
        try:
            for txt in params.strip().split(";"):
                if txt=="":
                    continue
                key,val=txt.split("=")
                params_dict[key]=eval(val)
        except:
            print("Parameter Parse Failure")
        self.params=self.default_params
        self.params.update(params_dict)

    def run(self,file_path):
        self.state=["Preparing",0]
        self.dataset=Dataset(file_path)
        self.dataset.open()
        self.data_info=self.dataset.get_data_info()
        T=self.data_info["T"]
        N_points=self.data_info["N_points"]
        anchor_labels=self.params["anchor_labels"]
        if anchor_labels is None:
            anchor_labels=np.arange(1,N_points+1)
        anchor_times=self.params["anchor_times"]
        if anchor_times is None:
            anchor_times=np.arange(1,T+1)
        points=self.dataset.get_points()
        dists=np.zeros((N_points,N_points))
        counts=np.zeros((N_points,N_points))
        self.state=["Calculating distance matrix",0]
        for i,points_t in enumerate(points):
            distmat=sspat.distance.squareform(sspat.distance.pdist(points_t[1:,:2]))
            valid=~np.isnan(distmat)
            np.fill_diagonal(valid, 0)
            distmat[~valid]=0
            dists+=distmat
            counts+=valid
            self.state[1]=int(100*((i+1)/T))
        dists=np.divide(dists,counts,where=counts!=0,out=np.full_like(dists,np.nan))

        self.state=["Assigning references",0]
        refss=[]
        for i in range(N_points):
            valids=~np.isnan(dists[i])
            if np.sum(valids)<3:
                refss.append(None)
            else:
                inds=np.nonzero(valids)[0]
                dists_=dists[i][inds]
                inside=dists_<self.params["radius"]
                if np.sum(inside)<3:
                    refss.append(None)
                else:
                    refss.append(inds[inside])
        for i in range(N_points):
            print(i,refss[i])
        self.state=["Locating Points",0]
        ptss=np.full_like(points,np.nan)
        for t in range(T):
            todo=np.nonzero(np.isnan(points[t,1:,0]))[0]
            for i in todo:
                refs=refss[i]
                if refs is None:
                    continue
                refcoords=points[t,refs+1,:2]
                validrefs=~np.isnan(refcoords[:,0])
                if validrefs.sum()<3:
                    continue
                refcoords=refcoords[validrefs]
                refds=dists[i,refs][validrefs]
                A=2*(refcoords[0][None,:]-refcoords[1:])
                b=refds[1:]**2-np.sum(refcoords[1:]**2,axis=1)-refds[0]**2+np.sum(refcoords[0]**2)
                #coord=np.linalg.inv(A)@b
                coord=np.linalg.inv((A.T)@A)@(A.T)@b
                ptss[t,1+i,:2]=coord
                ptss[t,1+i,2]=0
            self.state[1]=int(100*((t+1)/T))
        self.dataset.set_data("helper_NPS",ptss,overwrite=True)
        self.dataset.close()
        self.state="Done"

    def quit(self):
        pass

class KernelCorrelation2D():
    default_params={"forward":True,"kernel_size":51,"search_size":101,"refine_size":3}
    help=str(default_params)
    def __init__(self,params):
        self.state=""
        self.cancel=False

        params_dict={}
        try:
            for txt in params.strip().split(";"):
                if txt=="":
                    continue
                key,val=txt.split("=")
                params_dict[key]=eval(val)
        except:
            print("Parameter Parse Failure")
        self.params=self.default_params
        self.params.update(params_dict)

    def run(self,file_path):
        import torch
        device="cuda" if torch.cuda.is_available() else "cpu"
        self.state=["Preparing",0]
        kernel_size=self.params["kernel_size"]
        search_size=self.params["search_size"]
        refine_size=self.params["refine_size"]
        corr_size=search_size-kernel_size+1

        self.state=["Preparing",0]
        self.dataset=Dataset(file_path)
        self.dataset.open()
        self.data_info=self.dataset.get_data_info()
        C=self.data_info["C"]
        T=self.data_info["T"]
        assert self.data_info["D"]==1,"Data must be 2d"

        grid=np.arange(kernel_size)-kernel_size//2
        gridc,gridx,gridy=np.meshgrid(np.arange(C),grid,grid,indexing="ij")
        grid=np.arange(search_size)-search_size//2
        sgridc,sgridx,sgridy=np.meshgrid(np.arange(C),grid,grid,indexing="ij")
        grid=np.arange(corr_size)-corr_size//2
        cgridx,cgridy=np.meshgrid(grid,grid,indexing="ij")
        grid=np.arange(refine_size)-refine_size//2
        rgridx,rgridy=np.meshgrid(grid,grid,indexing="ij")
        rgrid=np.stack([rgridx,rgridy],axis=0)


        ptss=self.dataset.get_points()
        self.state=["Running",0]
        for i in range(T-1):
            if self.cancel:
                self.quit()
                return
            t_now=i
            t_next=i+1
            if i==0:
                im_now=self.dataset.get_frame(t_now)
            else:
                im_now=im_next
            im_next=self.dataset.get_frame(t_next)

            valid=~np.isnan(ptss[t_now,:,0])
            labels=[]
            kernels=[]
            images=[]
            for label,valid in enumerate(valid):
                if not valid:
                    continue
                labels.append(label)
                coord=ptss[t_now,label,:2]
                kernels.append(sim.map_coordinates(im_now[:,:,:,0],[gridc,coord[0]+gridx,coord[1]+gridy],order=1)/255)
                images.append(sim.map_coordinates(im_next[:,:,:,0],[sgridc,coord[0]+sgridx,coord[1]+sgridy],order=1)/255)
            kernels=np.stack(kernels,0)
            kernels-=kernels.mean(axis=(2,3))[:,:,None,None]
            kernels/=(kernels.std(axis=(2,3))[:,:,None,None]+1e-10)
            images=np.stack(images,0)
            images-=images.mean(axis=(2,3))[:,:,None,None]
            images/=(images.std(axis=(2,3))[:,:,None,None]+1e-10)

            corrs=self.get_corrs(images,kernels,device=device)
            disps=self.get_disps(corrs,rgrid)

            for label,disp in zip(labels,disps):
                if np.isnan(ptss[t_next,label,0]):
                    ptss[t_next,label,:2]=ptss[t_now,label,:2]+disp
                    ptss[t_next,label,2]=0
            self.state[1]=int(100*((i+1)/(T-1)))
            if i==500:
                break

        self.dataset.set_data("helper_KernelCorrelation2D",ptss,overwrite=True)
        self.dataset.close()
        self.state="Done"


    def get_corrs(self,images,kernels,device="cpu"):
        import torch
        B,C,W,H=images.shape
        ten_ker=torch.tensor(kernels,dtype=torch.float32,device=device)
        ten_im=torch.tensor(images,dtype=torch.float32,device=device).reshape(1,B*C,W,H)
        corrs=torch.nn.functional.conv2d(ten_im,ten_ker,groups=B)
        return corrs.reshape(B,C,corrs.shape[2],corrs.shape[3])

    def get_disps(self,corrs,rgrid):
        corrs=corrs.sum(1)
        B,W,H=corrs.shape
        corrs=corrs.reshape(B,-1).cpu().numpy()
        maxinds=np.argmax(corrs,axis=1)
        peak_coords=np.stack(np.unravel_index(maxinds,(W,H)),axis=1)
        #s=rgrid.shape[1]//2
        #valids=(peak_coords[:,0]>=s)*(peak_coords[:,0]<(W-s))*(peak_coords[:,1]>=s)*(peak_coords[:,1]<(H-s))
        #peak_coords_float=peak_coords.astype(np.float32)
        #for i,(peak_coord,valid) in enumerate(zip(peak_coords,valids)):
        #    if valid:
        #        weight=np.clip(corrs[peak_coord[0]+rgrid[0],peak_coord[1]+rgrid[1]],0,np.inf)
        #        if np.sum(weight)!=0:
        #            peak_coords_float[i]+=np.sum(weight[None]*rgrid,axis=(1,2))/np.sum(weight)
        return peak_coords-W//2

    def quit(self):
        self.dataset.close()

class InvDistInterp():
    default_params={"t_ref":0,"epslion":1e-10}
    help=str(default_params)
    def __init__(self,params):
        self.state=""
        self.cancel=False

        params_dict={}
        try:
            for txt in params.strip().split(";"):
                if txt=="":
                    continue
                key,val=txt.split("=")
                params_dict[key]=eval(val)
        except:
            print("Parameter Parse Failure")
        self.params=self.default_params
        self.params.update(params_dict)

    def run(self,file_path):
        epsilon=self.params["epslion"]
        t_ref=self.params["t_ref"]

        self.state=["Preparing",0]
        self.dataset=Dataset(file_path)
        self.dataset.open()
        self.data_info=self.dataset.get_data_info()
        T=self.data_info["T"]
        N_points=self.data_info["N_points"]
        points=self.dataset.get_points()
        dists=np.zeros((N_points,N_points))
        counts=np.zeros((N_points,N_points))
        self.state=["Calculating distance matrix",0]
        """
        for i,points_t in enumerate(points):
            distmat=sspat.distance.squareform(sspat.distance.pdist(points_t[1:,:2]))
            valid=~np.isnan(distmat)
            np.fill_diagonal(valid, 0)
            distmat[~valid]=0
            dists+=distmat
            counts+=valid
            self.state[1]=int(100*((i+1)/T))
        dists=np.divide(dists,counts,where=counts!=0,out=np.full_like(dists,np.nan))
        """
        points_ref=points[t_ref,1:,:2]
        existing_ref=~np.isnan(points_ref[:,0])
        distmat=sspat.distance.squareform(sspat.distance.pdist(points_ref))
        np.fill_diagonal(distmat,np.nan)

        self.state=["Locating Points",0]
        ptss=np.full_like(points,np.nan)
        for t in range(T):
            if self.cancel:
                self.quit()
                return
            existing=~np.isnan(points[t,1:,0])
            if existing.sum()==0:
                pass
            todo=np.nonzero(~(existing*existing_ref))[0]
            for i in todo:
                validrefs=np.nonzero((~np.isnan(distmat[i]))*existing*existing_ref)[0]
                if len(validrefs)==0:
                    continue
                weights=(1/(distmat[i,validrefs]+epsilon))**2
                vecs=points[t,validrefs+1,:2]-points_ref[validrefs,:2]
                vec=(weights[:,None]*vecs).sum(0)/weights.sum()
                ptss[t,1+i,:2]=points_ref[i]+vec
                ptss[t,1+i,2]=0

            self.state[1]=int(100*((t+1)/T))
        self.dataset.set_data("helper_InvDistInterp",ptss,overwrite=True)
        self.dataset.close()
        self.state="Done"

    def quit(self):
        self.dataset.close()
        pass

def run(name,command_pipe_sub,file_path,params):
    method=getattr(currmod,name)(params)
    thread=threading.Thread(target=method.run,args=(file_path,))
    while True:
        command=command_pipe_sub.recv()
        if command=="run":
            thread.start()
        elif command=="report":
            command_pipe_sub.send(method.state)
        elif command=="cancel":
            method.cancel=True
            thread.join()
            break
        elif command=="close":
            thread.join()
            break

methodnames=["NN","InvDistInterp"]
methodhelps={}
for name in methodnames:
    methodhelps[name]=getattr(currmod,name).help


if __name__=="__main__":
    import sys
    fp=sys.argv[1]
    method=NN("Targeted=True;num_additional=3")
    method.run(fp)
