#### Import needed libraries
import sys
import os
import h5py
import importlib
import glob
import numpy as np
import torch
import NNtools
import shutil
import multiprocessing
import scipy.spatial as spat
import time

#### This is the grammar to parse the command line parameters
sys.path.append("src/neural_network_scripts/models")
assert len(sys.argv)==3 or len(sys.argv)==4
## Don't change
logfn=sys.argv[2]
dataset_path=sys.argv[1]
dataset_name=os.path.split(dataset_path)[1].split(".")[0]
props=dataset_name.split("_")
NetName=props[-2]
runname=props[-1]
if len(sys.argv)==4:
    runname=sys.argv[3]
identifier="net/"+NetName+"_"+runname



#### These are the run options
########################################################################

####
#Most frequent  options
adiabatic_trick=False
deformation_trick=True
num_epochs=800
#####

##Run details
verbose=True
requeue=False
if requeue:
    usenet="net"
skiptrain=False
from_points=True
get_points=True
if from_points:
    thres=4
if from_points or get_points:
    distthres=4
#Data side run detail
channel_num=2
complete=False #Whether each annotated frame is completely annotated
if from_points:
    min_num_for_mask=15
##directory handling
reusedirec="data/data_temp/zmdir"
#purely computational parameters
if get_points:
    chunksize=30#for pointdat
#neural network training parameters
batch_size=3
aug_dict={100:"aff_cut_four"} #BENCH
#aug_dict={0:"none"} #BENCH
lr=0.0003
patience=8
num_workers=10
print_iou_every=10
tnum="all"
vnum=0
traininds_given=False
if traininds_given:
    traininds=[]
    valinds=[]#indices should be given as list or 1d numpy array
if deformation_trick:
    num_additional=80
    deformation_num_epochs=80
    deformation_augdict={0:"aff_cut"}

if adiabatic_trick:
    assert not requeue,"requeue not supported with adiabatic trick"
    #how to get k
    k_by_func=True
    if k_by_func:
        def k_func():
            k=list(np.linspace(3,25,150).astype(np.int16))
            while np.sum(k)<T:
                k.append(25)
            return np.array(k)
    else:
        k=6#number or numbers to feed in at once, can be array or int
    #memory of the chain
    short_memory=2 #nearby memory to train in
    num_random_memory=2 #randomly added memory
    num_random_memory_train=1 #randomly added memory from original training data(potential overlap with above)
    #training parameters
    #dset size will be k*short_memory+num_random_memory+num_random_memory_train except for edge cases
    def adiabatic_epoch_func(iters):#function giving epochs for adiabatic trick
        min_adia_epochs=4
        return max(100//iters,min_adia_epochs)
    adiabatic_aug_dict={1:"aff_cut"}
    lr_adia=0.0003
########################################################################

####Check the dependencies
dependencies=["W","H","D","C","T","N_neurons"]
h5=h5py.File(dataset_path,"r+")
for dep in dependencies:
    if dep not in h5.attrs.keys():
        h5.close()
        assert False, "Dependency "+dep+" not in  attributes"
if deformation_trick:
    assert "distmat" in h5.keys(), "no distmat given but deformation requested"
T=h5.attrs["T"]
C,W,H,D=h5.attrs["C"],h5.attrs["W"],h5.attrs["H"],h5.attrs["D"]#x,y,z ordering
channel_num=min(channel_num,C)
shape=(channel_num,W,H,D)#x,y,z ordering
gridpts=np.moveaxis(np.array(np.meshgrid(np.arange(W),np.arange(H),np.arange(D),indexing="ij")),0,3)
grid=gridpts.reshape(-1,3)



#### Extra parameters for adiabatic_trick
if adiabatic_trick:
    if k_by_func:
        k=k_func()
    elif type(k)==int:
        nums=(T//k+1)
        k=np.full(nums,k)
    assert np.sum(k)>=T
    #default method to get the lineup
    def get_lineup_def():
        ts=np.arange(T)
        ds=np.min(np.abs(ts[:,None]-np.array(traininds)[None,:]),axis=1)
        return np.argsort(ds)
    # default method to get the points, may include a checking procedure
    def get_pts(i):
        pts,_=NNtools.get_pts_iou(NNtools.pack(h5,identifier,i,gridpts,num_classes,thres=thres))
        pts=pts.astype(np.float32)
        return pts
    #update a mask prediction
    def updatemask(i,pts):
        key=identifier+"/"+str(i)+"/pts"
        if key in h5.keys():
            del h5[key]
        dset=h5.create_dataset(key,(num_classes,3),dtype="f4")
        dset[...]=pts
        fr=np.array(h5[str(i)+"/frame"])
        mask=NNtools.get_mask(fr[0],pts,num_classes,grid,thres=thres,distthres=distthres).astype(np.int16)
        h5[identifier+"/"+str(i)+"/predmask"][...]=mask



#### logging file function, this is the runtime log
def write_log(txt,end="\n"):
    with open(logfn,"w") as f:
        f.write(txt+end)
logform="Prepare={:04.4f} Train={:04.4f} Predict={:04.4f} GetPoints={:04.4f}"
write_log(logform.format(0.,0.,0.,0.))



#### saving backup function
def repack():
    global h5
    h5.close()
    NNtools.repack(dataset_path)
    h5=h5py.File(dataset_path,"r+")
def save_backup():
    global h5
    h5.close()
    shutil.copyfile(dataset_path,dataset_path+"_backup")
    h5=h5py.File(dataset_path,"r+")



############Handle run related parameters############
write_log(logform.format(0.,0.,0.,0.))
if requeue:
    assert identifier in h5.keys(), "requeue requested but identifier not present"
    log=h5[identifier].attrs["log"]
else:
    if identifier in h5.keys():
        del h5[identifier]
    h5.create_group(identifier)
    log=""
    h5[identifier].attrs["log"]=log



############setup the run directory############
if reusedirec is not None:
    datadir=reusedirec
else:
    datadir=os.path.join("data","data_temp",dataset_name)



############Clean up the pointdat for times with more than some number of points, also reject the remaining classes to save memory############
if verbose:
    print("Preparing...")
try:
    #### get basic parameters
    if from_points:
        pointdat=np.array(h5["pointdat"])
        existing=np.logical_not(np.isnan(pointdat[:,:,0]))
        pointdat[np.sum(existing,axis=1)<min_num_for_mask,:,:]=np.nan
        existing=np.logical_not(np.isnan(pointdat[:,:,0]))
        existing_classes=np.any(existing,axis=0)
        num_classes=np.max(np.nonzero(existing_classes)[0])+1
        pointdat=pointdat[:,:num_classes,:]
        pts_exists=np.any(np.logical_not(np.isnan(pointdat[:,:,0])),axis=1)
    else:
        num_classes=h5.attrs["N_neurons"]+1



############Unpack h5 if not already unpacked ############
    if reusedirec is None or not os.path.exists(reusedirec):
        os.mkdir(datadir)
        os.mkdir(os.path.join(datadir,"frames"))
        os.mkdir(os.path.join(datadir,"highs"))
        os.mkdir(os.path.join(datadir,"masks"))
        Ntot=0
        Nans=0
        for i in range(T):
            write_log(logform.format(min(i/T,0.8),0.,0.,0.))
            fr=np.array(h5[str(i)+"/frame"]).astype(np.int16)
            np.save(os.path.join(datadir,"frames","frame_"+str(i)+".npy"),fr)
            Ntot+=1
            if str(i)+"/high" in h5.keys():
                np.save(os.path.join(datadir,"highs","high_"+str(i)+".npy"),np.array(h5[str(i)+"/high"]).astype(np.int16))
            else:
                np.save(os.path.join(datadir,"highs","high_"+str(i)+".npy"),np.full((1,W,H),255).astype(np.int16))
            if from_points:
                if pts_exists[i]:
                    mask=NNtools.get_mask(fr[0],pointdat[i],num_classes,grid,thres=thres,distthres=distthres).astype(np.int16)
                    np.save(os.path.join(datadir,"masks","mask_"+str(i)+".npy"),mask)
                    Nans+=1
            elif str(i)+"/mask" in h5.keys():
                np.save(os.path.join(datadir,"masks","mask_"+str(i)+".npy"),np.array(h5[str(i)+"/mask"]).astype(np.int16))
                Nans+=1
        assert Nans>0, "At least one mask is needed"



############Initialize the network ############
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log+="device= "+str(device)+"\n"
    NetMod = importlib.import_module(NetName)
    net=NetMod.Net(n_channels=shape[0],num_classes=num_classes)
    if requeue:
        net.load_state_dict(NNtools.load_from_h5(h5[identifier+"/"+usenet]))
        log+="weight load Successful\n"
        if verbose:
            print("weight load Successful\n")
    net.to(device=device)
    n_params=sum([p.numel() for p in net.parameters()])
    log+="Total number of parameters:"+str(n_params)+"\n"


############Initialize Dataset and Dataloaders############
    allset=NNtools.TrainDataset(datadir,shape,high=False)
    totnum=len(allset)
    if traininds_given:
        tindices=allset.real_ind_to_dset_ind(traininds)
        tsampler=torch.utils.data.SubsetRandomSampler(tindices)
        traindataloader= torch.utils.data.DataLoader(allset, batch_size=batch_size,sampler=tsampler, num_workers=num_workers,pin_memory=True)
        if len(valinds)!=0:
            vindices=allset.real_ind_to_dset_ind(valinds)
            vsampler=torch.utils.data.SubsetRandomSampler(vindices)
            valdataloader=torch.utils.data.DataLoader(allset,batch_size=batch_size,sampler=vsampler,num_workers=num_workers,pin_memory=True)
            tnum,vnum=len(traininds),len(valinds)
        else:
            valdataloader=None
            tnum=len(traininds)
            vnum=0
    elif tnum=="all" or vnum==0:#vnum should exist if tnum!="all"
        traindataloader= torch.utils.data.DataLoader(allset, batch_size=batch_size,shuffle=True, num_workers=num_workers,pin_memory=True)
        valdataloader=None
        traininds=allset.indlist
        tnum=len(allset)
        vnum=0
    else:
        if totnum==(tnum+vnum):
            tset,vset=torch.utils.data.random_split(allset,[tnum,vnum])
        else:
            tset,vset,_=torch.utils.data.random_split(allset,[tnum,vnum,totnum-tnum-vnum])
        traindataloader= torch.utils.data.DataLoader(tset, batch_size=batch_size,shuffle=True, num_workers=num_workers,pin_memory=True)
        valdataloader=torch.utils.data.DataLoader(vset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True)
        traininds=allset.indlist[tset.indices]
    #### requeue should match training indices
    if requeue and not adiabatic_trick:
        assert all(np.array(h5[identifier].attrs["traininds"])==traininds),"traininds not matching"
    #### save training indices
    h5[identifier].attrs["traininds"]=traininds
    log+="Training with: trainset: "+str(tnum)+" valset: "+str(vnum)+"\n"
    num_trains=len(traindataloader)
    digits=len(str(num_trains))#for pretty print
    if valdataloader is None:
        num_vals=None
        digits_v=None
    else:
        num_vals=len(valdataloader)
        digits_v=len(str(num_vals))



############define the criterion function and optimizers############
    if complete:
        def criterion(pred_raw,target_mask,fr):
            return torch.nn.functional.cross_entropy(pred_raw,target_mask)
    else:
        def criterion(pred_raw,target_mask,fr):
            return NNtools.selective_ce(pred_raw,target_mask)

    optimizer=torch.optim.Adam(net.parameters(),lr=lr,amsgrad=True)
    scheduler=None#torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.3,patience=patience,min_lr=5e-5)



############Prepare logging############
    #copy the log if we are on requeue
    h5[identifier].attrs["log"]=log
    if requeue:
        loss_iou=np.array(h5[identifier]["loss_iou"])
        inds=list(loss_iou[:,0])
        losses=list(loss_iou[:,1])
        iouss=[ious for ious in loss_iou[:,2:]]
    else:
        losses=[]
        iouss=[]
        inds=[]



########################################################################
# Start of Training
    log+="Starting Training\n\n"
    if skiptrain:
        print("Skipping Training")
    else:
        if verbose:
            print("Starting Train")

        #do standard training
        NNtools.train(h5,identifier,device,net,optimizer,criterion,\
                NNtools.get_ious,scheduler,allset,traindataloader,num_epochs,aug_dict,\
                log,verbose,losses,iouss,inds,print_iou_every,digits,num_trains,vnum=vnum,\
                valdataloader=valdataloader,digits_v=digits_v,num_vals=num_vals,write_log=True,\
                logfn=logfn)

        #save net in h5
        if "basenet" in h5[identifier].keys():
            del h5[identifier]["basenet"]
        h5.create_group(identifier+"/basenet")
        NNtools.save_into_h5(h5[identifier+"/basenet"],net.state_dict())
        #repack()
        #save_backup()#make h5 backup

        if deformation_trick:
            if verbose:
                print("Adding Deformed Frames")

            #copy the existing masks first
            dir_deformations=os.path.join(datadir,"deformations")
            if os.path.exists(dir_deformations):
                shutil.rmtree(dir_deformations)
            os.mkdir(dir_deformations)
            os.mkdir(os.path.join(datadir,"deformations","frames"))
            os.mkdir(os.path.join(datadir,"deformations","masks"))
            for name in np.array(allset.filelist)[allset.real_ind_to_dset_ind(traininds)]:
                i=int(os.path.split(name)[-1].split(".")[0].split("_")[1])
                shutil.copyfile(os.path.join(datadir,"frames","frame_"+str(i)+".npy"),os.path.join(datadir,"deformations","frames","frame_"+str(i)+".npy"))
                shutil.copyfile(name,os.path.join(datadir,"deformations","masks","mask_"+str(i)+".npy"))



            distmat=np.array(h5["distmat"])
            additional_inds=NNtools.select_additional(T,traininds,distmat,num_additional)[len(traininds):]
            evalset=NNtools.EvalDataset(datadir,shape)
            for i in additional_inds:
                with torch.no_grad():
                    fr,mask=evalset[i]
                    fr=fr.unsqueeze(0).to(device=device, dtype=torch.float32)
                    pred=net(fr)
                    predmask=torch.argmax(pred[0],dim=0).cpu().detach().numpy().astype(np.int16)

                    fr_r=np.clip(fr[0,0].cpu().detach().numpy()*255,0,255).astype(np.int16)#zero'th channel weighting
                    pts_child,_=NNtools.get_pts_iou( (fr_r,predmask,None,gridpts,num_classes,thres,False) )
                    pts_child=pts_child.astype(np.float32)

                    """
                    import matplotlib.pyplot as plt
                    plt.subplot(3,2,1)
                    plt.imshow(np.max(fr_r,axis=2).T)
                    plt.title(str(i))
                    plt.scatter(pts_child[:,0],pts_child[:,1],c="w",s=1)
                    plt.subplot(3,2,2)
                    plt.imshow(np.max(predmask,axis=2).T,cmap="nipy_spectral",interpolation="none")
                    """

                i_parent=traininds[np.argmin(distmat[traininds,i])]
                pts_parent=pointdat[i_parent]

                deformation,_,_,success=NNtools.get_deformation(ptfrom=pts_parent,ptto=pts_child,sh=(W,H,D),k_cut_dimless=2.5,scale=(1,1,5),device=device)
                if success!=0:
                    print("Deformation failed due to ",success)
                    continue
                fr,mask=evalset[i_parent]
                """
                plt.subplot(3,2,3)
                plt.imshow(np.max(fr[0].cpu().detach().numpy(),axis=2).T)
                plt.title(str(i_parent))
                plt.colorbar()
                plt.subplot(3,2,4)
                plt.imshow(np.max(mask.cpu().detach().numpy(),axis=2).T,cmap="nipy_spectral",interpolation="none")
                """

                fr=fr.unsqueeze(0).to(device=device, dtype=torch.float32)
                mask=mask.unsqueeze(0).to(device=device, dtype=torch.long)
                fr,mask=NNtools.deform((W,H,D),deformation,fr,mask=mask)
                fr=np.clip(fr[0].cpu().detach().numpy()*255,0,255).astype(np.int16)
                mask=mask[0].cpu().detach().numpy().astype(np.int16)

                """
                plt.subplot(3,2,5)
                plt.imshow(np.max(fr[0],axis=2).T)
                plt.title(str(distmat[i_parent,i]))
                plt.colorbar()
                plt.subplot(3,2,6)
                plt.imshow(np.max(mask,axis=2).T,cmap="nipy_spectral",interpolation="none")
                plt.show()
                """

                np.save(os.path.join(datadir,"deformations","frames","frame_"+str(T+i)+".npy"),fr) #we add T to avoid collision, but don't want to define another class
                np.save(os.path.join(datadir,"deformations","masks","mask_"+str(T+i)+".npy"),mask)

            #now add the new masks
            allset=NNtools.TrainDataset(os.path.join(datadir,"deformations"),shape,high=False)
            traindataloader= torch.utils.data.DataLoader(allset, batch_size=batch_size,shuffle=True, num_workers=num_workers,pin_memory=True)

            optimizer=torch.optim.Adam(net.parameters(),lr=lr)
            NNtools.train(h5,identifier,device,net,optimizer,criterion,\
                    NNtools.get_ious,None,allset,traindataloader,deformation_num_epochs,deformation_augdict,\
                    log,verbose,losses,iouss,inds,print_iou_every,digits,len(traindataloader),vnum=0,\
                    valdataloader=None,digits_v=None,num_vals=None,write_log=True,\
                    logfn=logfn)


        if adiabatic_trick:#more training steps for the trick
            #now call lineup of the frames
            if "lineup" in h5.keys():
                lineup=np.array(h5["lineup"])
            else:#if non existing, make the default lineup
                lineup=get_lineup_def()
                dset=h5.create_dataset("lineup",shape=(T,),dtype="i2")
                dset[...]=np.array(lineup).astype(np.int16)
            assert all((np.sort(np.array(traininds))==np.sort(lineup[:len(traininds)]))), "The first lineup elements should be traininds"
            repack()
            save_backup()#make h5 backup

            dir_predmasks=os.path.join(datadir,"predmasks")
            if os.path.exists(dir_predmasks):
                shutil.rmtree(dir_predmasks)
            os.mkdir(dir_predmasks)
            for name in np.array(allset.filelist)[allset.real_ind_to_dset_ind(traininds)]:
                shutil.copyfile(name,os.path.join(datadir,"predmasks",os.path.split(name)[-1]))

            evalset=NNtools.EvalDataset(datadir,shape)#we need the evalutation set for aligned indices for all frames
            lineup=list(lineup)[len(traininds):]#traininds should be at the begining
            added=[list(traininds)]#traininds are already added
            c=0
            while True:
                #add mask
                ichilds=[]
                #pop k[c] elemets from lineup
                k_one=k[c]
                for _ in range(k_one):
                    if len(lineup)==0:
                        break
                    ichilds.append(lineup.pop(0))
                added.append(ichilds)
                if verbose:
                    print(len(lineup)," left")
                #make predictions
                net.eval()
                for ichild in ichilds:
                    with torch.no_grad():
                        fr,mask=evalset[ichild]
                        fr=fr.unsqueeze(0).to(device=device, dtype=torch.float32)
                        pred=net(fr)
                        predmask=torch.argmax(pred[0],dim=0).cpu().detach().numpy().astype(np.int16)
                        del fr,pred
                    if identifier+"/"+str(ichild)+"/predmask" in h5.keys():
                        del h5[identifier+"/"+str(ichild)+"/predmask"]
                    h5[identifier].create_dataset(str(ichild)+"/predmask",(predmask.shape),dtype="i2",compression="gzip")
                    h5[identifier][str(ichild)+"/predmask"][...]=predmask

                    #This re-masks the data
                    ptschild=get_pts(ichild)
                    updatemask(ichild,ptschild)
                    np.save(os.path.join(datadir,"predmasks","mask_"+str(ichild)+".npy"),np.array(h5[identifier+"/"+str(ichild)+"/predmask"]).astype(np.int16))

                # regen optimizers
                optimizer=torch.optim.Adam(net.parameters(),lr=lr_adia,amsgrad=True)
                #update dataset
                allset=NNtools.TrainDataset(datadir,shape,maskdirname="predmasks")
                # break if we are done: case if np.cumsum(k) exactly ends at lT
                if len(allset)==T:
                    break
                # training phase
                #all available training indices
                t_inds_all=[el for els in added for el in els]
                #those in memory
                t_inds_memory=[el for els in added[-short_memory:] for el in els]#last in memory
                t_inds_memory_dict=dict(zip(t_inds_memory,[True for _ in range(len(t_inds_memory))]))
                #those not in memory
                buff=[]
                for t in t_inds_all:
                    if t not in t_inds_memory_dict.keys():
                        buff.append(t)
                #get epochs
                ad_epochs=adiabatic_epoch_func(len(t_inds_memory))
                for epoch in range(ad_epochs):
                    # random select random memory
                    t_inds_random_memory=list(np.random.choice(buff,min(num_random_memory,len(buff)),replace=False))
                    # random select random memory train
                    t_inds_random_memory_train=list(np.random.choice(traininds,min(num_random_memory_train,len(traininds)),replace=False))
                    #all tinds
                    tinds=[*t_inds_memory,*t_inds_random_memory,*t_inds_random_memory_train]
                    tset=torch.utils.data.Subset(allset,allset.real_ind_to_dset_ind(tinds))
                    traindataloader= torch.utils.data.DataLoader(tset, batch_size=batch_size,shuffle=True, num_workers=num_workers,pin_memory=True)
                    num_trains=len(traindataloader)
                    if epoch in adiabatic_aug_dict.keys():
                        text="augment is now:"+allset.change_augment(adiabatic_aug_dict[epoch])
                        log+=text
                        if verbose:
                            print(text)
                    #num epoch used is 1, we explicilty use a super epoch.
                    NNtools.train(h5,identifier,device,net,optimizer,criterion,NNtools.get_ious,None,allset,traindataloader,1,\
                        {},log,verbose,losses,iouss,inds,print_iou_every,digits,\
                        num_trains,vnum=0,valdataloader=None,digits_v=None,num_vals=None)
                    #CRITICAL, emergency break
                    if os.path.exists("STOP"):
                        break
                if os.path.exists("STOP"):
                    break
                c+=1
                if c%20==0:
                    repack()
                    save_backup()#save backup, harvard rc cluster
# End of Training
########################################################################
    #save log
    h5[identifier].attrs["log"]=log

    ##Prediction phase
    evalset=NNtools.EvalDataset(datadir,shape)
    iouss=np.full((T,num_classes),np.nan)

    #predict all masks
    net.eval()
    for i in range(T):
        if verbose:
            print("Evaluating "+str(i)+"/"+str(T))
        with torch.no_grad():
            fr,mask=evalset[i]
            fr=fr.unsqueeze(0).to(device=device, dtype=torch.float32)
            pred=net(fr)
            if mask is not None:
                mask=mask.unsqueeze(0).to(device=device, dtype=torch.long)
                ious=NNtools.get_ious(pred,mask,False)
            else:
                ious=np.full(num_classes,np.nan)
            iouss[i]=ious
            predmask=torch.argmax(pred[0],dim=0).cpu().detach().numpy().astype(np.int16)
        if identifier+"/"+str(i)+"/predmask" in h5.keys():
            del h5[identifier+"/"+str(i)+"/predmask"]
        h5[identifier].create_dataset(str(i)+"/predmask",(predmask.shape),dtype="i2",compression="gzip")
        h5[identifier][str(i)+"/predmask"][...]=predmask
        write_log(logform.format(1.,1.,i/T,0.))

    #calculate prediction iou
    if "pred_iou" in h5[identifier].keys():
        del h5[identifier]["pred_iou"]
    dset=h5[identifier].create_dataset("pred_iou",(T,num_classes),dtype="f4",compression="gzip")
    dset[...]=np.array(iouss).astype(np.float32)

    #report end of prediction
    if verbose:
        print("Prediction Complete.\n")
    log+="Prediction Complete\n\n"
    h5[identifier].attrs["log"]=log

    #get points if needed
    if get_points:
        if verbose:
            print("Start Point Extraction\n")
        log+="Start Point Extraction\n\n"
        ptss=np.full((T,h5.attrs["N_neurons"]+1,3),np.nan)
        ious=np.full((T,num_classes,num_classes),np.nan)
        pool=multiprocessing.Pool(multiprocessing.cpu_count())
        result=pool.imap(NNtools.get_pts_iou,(NNtools.pack(h5,identifier,i,gridpts,num_classes,thres) for i in range(T)),chunksize=chunksize)
        for idx,res in enumerate(result):
            print(str(idx)+"/"+str(T))
            pts,iou=res
            ptss[idx][:num_classes]=pts
            ious[idx]=iou
            write_log(logform.format(1.,1.,1.,idx/T))
        if "NN_pointdat" not in h5[identifier].keys():
            h5[identifier].create_dataset("NN_pointdat",ptss.shape,dtype="f4")
        h5[identifier]["NN_pointdat"][...]=ptss
        if "ious" not in h5[identifier].keys():
            h5[identifier].create_dataset("ious",ious.shape,dtype="i2")
        h5[identifier]["ious"][...]=ious
        write_log(logform.format(1.,1.,1.,.1))
        log+="Point Extraction Done\n\n"


    #report succesful run
    if verbose:
        print("Run Fully Successful\n")
    log+="Run Fully Successful\n"
    h5[identifier].attrs["log"]=log

    time.sleep(0.2)
    #repack h5 file
    if verbose:
        print("Repacking h5.")
    #repack()

    #remove used directory
    if reusedirec is None:
        shutil.rmtree(datadir)
    write_log(logform.format(1.,1.,1.,1.))

    if verbose:
        print("DONE")
except Exception as exception:
    if reusedirec is None:
        shutil.rmtree(datadir)
    raise exception
