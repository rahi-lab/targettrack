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

class NNTA():
    default_params={"min_points":5,"channels":[0],"mask_radius":4,

    "save_suff":"",
    "network":"ThreeDCN",
    "lr":0.01,
    "n_epochs":1000,"batch_size":3,
    "weight_channel":None,"use_path":None,"save_path":None,"save_suff":"",
    "fast_points":False,

    "Targeted":True,
    "recalcdistmat":False,"n_steps_posture":5000,"batch_size_posture":16,"umap_dim":2,
    "num_additional":20,
    "deformation_name":"lowk2d",
    "deformparams":{"k_cut_dimless":2.5,"lr":0.1,"iterations":1200,"frac":0.3,"lambda_div":1,"at_least":5},
    "verbose":False,
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

        if type(self.params["n_epochs"])==list:
            epochs=np.array(self.params["n_epochs"])
            self.params["n_epochs"]=np.max(epochs)
            self.params["peeks"]=list(epochs)

        assert self.params["min_points"]>0

    def run(self,file_path):
        from src.methods.neural_network_tools import NNtools_points
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

        #make files
        if True:
            self.state=["Making Files",0]
            T,N_points,C,W,H,D=self.data_info["T"],self.data_info["N_neurons"],self.data_info["C"],self.data_info["W"],self.data_info["H"],self.data_info["D"]

            points=self.dataset.get_points()
            min_points=self.params["min_points"]
            channels=self.params["channels"]
            if channels is None:
                channels=np.arange(C)
            n_channels=len(channels)
            grid=np.stack(np.meshgrid(np.arange(W),np.arange(H),np.arange(D),indexing="ij"),axis=0)
            gridpts=grid.reshape(3,-1).T
            grid_dimend=grid.transpose(1,2,3,0)

            traininds=[]
            train_data=NNtools_points.TrainDataset((n_channels,W,H,D))

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
                #if (self.params["traininds"] is not None) and (t not in self.params["traininds"]):
                #    continue
                pts=points[t]
                valid=~np.isnan(pts[:,0])
                if valid.sum()>=min_points:
                    inds=np.nonzero(valid)[0]
                    maskpts=NNtools_points.get_maskpts(inds_to_labels[inds],pts[inds],gridpts,self.params["mask_radius"])
                    mask=torch.tensor(maskpts.reshape(W,H,D),dtype=torch.uint8)
                    torch.save(mask,os.path.join(self.folpath,"masks",str(t)+".pt"))
                    train_data.add_data(image,mask,"gt")
                    traininds.append(t)
                self.state[1]=int(100*(t+1)/T)
                if self.params["verbose"]:
                    print("\r"+self.state[0]+" "+str(self.state[1]),end="")
            traininds=np.array(traininds)

        #Make posture space if TA
        if self.params["Targeted"] and ((not self.dataset.exists("distmat")) or self.params["recalcdistmat"]):
            self.state=["Embedding Posture Space Training",0]
            self.encnet=Networks.AutoEnc2d(sh2d=(W,H),n_channels=n_channels,n_z=min(20,T//4))
            self.encnet.to(device=self.device,dtype=torch.float32)
            self.encnet.train()

            data=NNtools_points.EvalDataset(folpath=self.folpath,T=T,maxz=True)
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
                    if self.params["verbose"]:
                        print("\r"+self.state[0]+" "+str(self.state[1]),end="")
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
                    if self.params["verbose"]:
                        print("\r"+self.state[0]+" "+str(self.state[1]),end="")
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
                vecs-=vecs.min(0)
                vecs/=vecs.max(0)
                self.dataset.set_data("umap_vecs",vecs,overwrite=True)
            distmat=sspat.distance_matrix(vecs,vecs).astype(np.float32)
            self.dataset.set_data("distmat",distmat,overwrite=True)

        #Train network
        if True:
            self.state=["Training Network",0]
            self.net=getattr(Networks,self.params["network"])(n_channels=n_channels,num_classes=N_labels)
            self.net.to(device=self.device,dtype=torch.float32)
            if self.params["use_path"] is not None:
                self.net.load_state_dict(torch.load(self.params["use_path"]))
            loader=torch.utils.data.DataLoader(train_data, batch_size=self.params["batch_size"],shuffle=True, num_workers=0,pin_memory=True)
            opt=torch.optim.Adam(self.net.parameters(),lr=self.params["lr"])
            n_epochs=self.params["n_epochs"]

            self.net.train()
            for epoch in range(n_epochs):
                for i,onebatch in enumerate(loader):
                    if self.cancel:
                        self.quit()
                        return
                    loss=NNtools_points.single_batch(onebatch,self.net,NNtools_points.selective_ce,opt,self.device)
                self.state[1]=int(100*(epoch/n_epochs))
                if self.params["verbose"]:
                    print("\r"+self.state[0]+" "+str(self.state[1]),end="")

        #Add TA frames and train
        if self.params["Targeted"]:
            #Make TA frames
            if True:
                self.net.eval()
                self.state=["Making TA",0]

                distmat=self.dataset.get_data("distmat")
                additional_inds=NNtools_points.get_additional_inds(traininds,distmat)
                evaldata=NNtools_points.EvalDataset(folpath=self.folpath,T=T,mask=True,maxz=False)

                n_added=0
                count=0
                while True:
                    if self.params["num_additional"]==0:
                        break
                    self.state[1]=int(100*((n_added)/self.params["num_additional"]))
                    if self.params["verbose"]:
                        print("\r"+self.state[0]+" "+str(self.state[1]),end="")

                    if count==len(additional_inds):
                        break
                    i_tar=additional_inds[count]
                    i_gt=traininds[np.argmin(distmat[traininds,i_tar])]
                    count+=1

                    #getting points
                    with torch.no_grad():
                        im=evaldata[i_tar][0].to(device=self.device,dtype=torch.float32)
                        maskpred=torch.argmax(self.net(im.unsqueeze(0))[0],dim=0).cpu().detach().numpy()

                        if self.params["fast_points"]:
                            pts_tar=np.full((N_points+1,3),np.nan)
                            nonback=maskpred>0
                            if np.sum(nonback)!=0:
                                pts=NNtools_points.get_point_fast3D(maskpred[nonback],grid_dimend[nonback],N_labels)
                                pts_tar[labels_to_inds]=pts
                        else:
                            if self.params["weight_channel"] is None:
                                pts_dict=NNtools_points.get_pts_dict(maskpred,grid,weight=np.ones(grid.shape[1:]).astype(np.float32))
                            else:
                                pts_dict=NNtools_points.get_pts_dict(maskpred,grid,weight=im[self.params["weight_channel"]].cpu().detach().numpy())
                            pts_tar=np.full((N_points+1,3),np.nan)
                            for label,coord in pts_dict.items():
                                pts_tar[labels_to_inds[label]]=coord
                    pts_gt=points[i_gt]

                    deformation,success=Deformation.get_deformation(name=self.params["deformation_name"],ptfrom=pts_gt,ptto=pts_tar,sh=(W,H,D),device=self.device,**self.params["deformparams"])
                    if not success:
                        continue
                    with torch.no_grad():
                        im,mask=evaldata[i_gt]
                        im=im.to(device=self.device,dtype=torch.float32)
                        mask=mask.to(device=self.device,dtype=torch.long)
                        im,mask=Deformation.deform(im,deformation,mask=mask)
                        im=torch.clip(im*255,0,255).to(device="cpu",dtype=torch.uint8).detach()
                        mask=mask.to(device="cpu",dtype=torch.uint8).detach()
                        train_data.add_data(im,mask,"ta")
                    n_added+=1
                    if n_added==self.params["num_additional"]:
                        break

            #print("training")
            #Train with TA frames
            if True:
                self.state=["Training TA",0]
                loader=torch.utils.data.DataLoader(train_data, batch_size=self.params["batch_size"],shuffle=True, num_workers=0,pin_memory=True)
                opt=torch.optim.Adam(self.net.parameters(),lr=self.params["lr"])
                self.net.train()
                for epoch in range(n_epochs):
                    for i,onebatch in enumerate(loader):
                        if self.cancel:
                            self.quit()
                            return
                        loss=NNtools_points.single_batch(onebatch,self.net,NNtools_points.selective_ce,opt,self.device)
                    self.state[1]=int(100*(epoch/n_epochs))
                    if self.params["verbose"]:
                        print("\r"+self.state[0]+" "+str(self.state[1]),end="")

        #Save Network
        if self.params["save_path"] is not None:
            torch.save(self.net.state_dict(),self.params["save_path"])

        #Extract points and save
        if True:
            self.state=["Extracting Points",0]
            ptss=np.full((T,N_points+1,3),np.nan)
            if self.params["weight_channel"] is None:
                weight=np.ones(grid.shape[1:]).astype(np.float32)
            evaldata=NNtools_points.EvalDataset(folpath=self.folpath,T=T,maxz=False)

            self.net.eval()
            with torch.no_grad():
                for i in range(T):
                    if self.cancel:
                        self.quit()
                        return
                    im=evaldata[i].to(device=self.device,dtype=torch.float32)
                    maskpred=torch.argmax(self.net(im.unsqueeze(0))[0],dim=0).cpu().detach().numpy()
                    if self.params["fast_points"]:
                        nonback=maskpred>0
                        if np.sum(nonback)!=0:
                            pts=NNtools_points.get_point_fast3D(maskpred[nonback],grid_dimend[nonback],N_labels)
                            ptss[i,labels_to_inds]=pts
                    else:
                        if self.params["weight_channel"] is None:
                            pts_dict=NNtools_points.get_pts_dict(maskpred,grid,weight=weight)
                        else:
                            im=im.cpu().detach().numpy()
                            pts_dict=NNtools_points.get_pts_dict(maskpred,grid,weight=im[self.params["weight_channel"]])
                        for label,coord in pts_dict.items():
                            ptss[i,labels_to_inds[label]]=coord
                    self.state[1]=int(100*((i+1)/T))
                    if self.params["verbose"]:
                        print("\r"+self.state[0]+" "+str(self.state[1]),end="")

            ptss[:,0,:]=np.nan

            name="NN"+("_" if self.params["save_suff"]!="" else "")+self.params["save_suff"]
            name=name.replace("TAb","TAb2")
            print("Saving as "+name)
            self.dataset.set_data("helper_"+name,ptss,overwrite=True)

        self.dataset.close()
        #shutil.rmtree(self.folpath)
        self.state="Done"

    def quit(self):
        pass
        #shutil.rmtree(self.folpath)

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


methodnames=["NNTA"]
methodhelps={}
for name in methodnames:
    methodhelps[name]=getattr(currmod,name).help

if __name__=="__main__":
    import sys
    fp=sys.argv[1]
    params=";".join(sys.argv[2:])
    print("Running with:")
    print(params)
    print()
    method=NNTA(params)
    method.run(fp)


