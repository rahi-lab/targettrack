#### Import needed libraries
import sys
import os
import h5py
import importlib
import numpy as np
import torch
import NNtools
import shutil
import time
import multiprocessing
import scipy.spatial as spat

#### This is the grammar to parse the command line parameters
sys.path.append("src/neural_network_scripts/models")
st=time.time()
assert len(sys.argv)==3
## Don't change
logfn=sys.argv[2]
dataset_path=sys.argv[1]
dataset_name=os.path.split(dataset_path)[1].split(".")[0]
props=dataset_name.split("_")
NetName=props[-2]
assert NetName=="RGN","Network mustk be RGN"
runname=props[-1]
identifier="net/"+NetName+"_"+runname

#### These are the run options
########################################################################
#Run status
adiabatic_trick=True
verbose=True
requeue=False
if requeue:
    usenet="net"
skiptrain=False
from_points=True
assert from_points==True,"For RGN we always start from points"
get_points=True
if from_points:
    distthres=4
if from_points or get_points:
    thres=4

#Run detail
channel_num=2
min_num_for_mask=40

##directory handling
reusedirec="data/data_temp/zmdir"

#purely computational parameters
chunksize=30#for pointdat

#neural network training parameters
n=1500
k_neighbor=100
dim_embed=1024
eval_rep=5
batch_size=10
num_epochs=3000
ep_augoff=400
lr=0.003
patience=10
num_workers=10
print_iou_every=10
tnum="all"
vnum=0
if adiabatic_trick:
    tnum="all"
    k_by_func=True
    if k_by_func:
        def k_func():
            k=list(np.linspace(4,25,150).astype(np.int16))
            while np.sum(k)<T:
                k.append(25)
            return np.array(k)
    else:
        k=6#number or numbers to feed in at once, can be array or int

    short_memory=2 #nearby memory to train in
    num_random_memory=2 #randomly added memory
    num_random_memory_train=1 #randomly added memory from original training data(potential overlap with above)

    #dset size will be k*short_memory+num_random_memory+num_random_memory_train except for edge cases
    def adiabatic_epoch_func(iters):#function giving epochs for adiabatic trick
        min_adia_epochs=4
        return max(80//iters,min_adia_epochs)
    adiabatic_ep_augoff=1
    lr_adia=0.0003
    patience_adia=2
########################################################################

####Check the dependencies
dependencies=["W","H","D","C","T","N_neurons"]
h5=h5py.File(dataset_path,"r+")
for dep in dependencies:
    if dep not in h5.attrs.keys():
        h5.close()
        assert False, "Dependency "+dep+" not in  attributes"
T=h5.attrs["T"]
C,W,H,D=min(channel_num,h5.attrs["C"]),h5.attrs["W"],h5.attrs["H"],h5.attrs["D"]#x,y,z ordering
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

    def get_lineup_def():
        ts=np.arange(T)
        ds=np.min(np.abs(ts[:,None]-np.array(traininds)[None,:]),axis=1)
        return np.argsort(ds)

    def updatemask(i,pts):
        key=identifier+"/"+str(i)+"/pts"
        if key in h5.keys():
            del h5[key]
        dset=h5.create_dataset(key,(num_classes,3),dtype="f4")
        dset[...]=pts

        fr=np.array(h5[str(i)+"/frame"])
        mask=NNtools.get_mask(fr[0],pts,num_classes,grid,thres=thres,distthres=distthres).astype(np.int16)
        key=identifier+"/"+str(i)+"/predmask"
        if key in h5.keys():
            del h5[key]
        dset=h5.create_dataset(key,mask.shape,dtype="i2",compression="gzip")
        dset[...]=mask

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

#### Handle run related parameters
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

if reusedirec is not None:
    datadir=reusedirec
else:
    datadir=os.path.join("data","data_temp",dataset_name)


#### The computation loop is all in a try except to delete the data
if verbose:
    print("Preparing...")
try:
    #### get basic parameters
    if from_points:
        pointdat=np.array(h5["pointdat"])
        #checked=np.array(h5["checked"])
        #checked=np.array([checked,checked,checked]).transpose(1,2,0)
        #pointdat=np.where(checked,pointdat,np.nan)
        existing=np.logical_not(np.isnan(pointdat[:,:,0]))
        pointdat[np.sum(existing,axis=1)<min_num_for_mask,:,:]=np.nan
        existing=np.logical_not(np.isnan(pointdat[:,:,0]))
        existing_classes=np.any(np.logical_not(np.isnan(pointdat[:,:,0])),axis=0)
        num_classes=np.max(np.nonzero(existing_classes)[0])+1
        pointdat=pointdat[:,:num_classes,:]
        pts_exists=np.any(np.logical_not(np.isnan(pointdat[:,:,0])),axis=1)
    else:
        num_classes=h5.attrs["N_neurons"]+1


    ####Unpack for fast, multiprocessed loading, unless already unpacked
    if reusedirec is None or not os.path.exists(reusedirec):
        os.mkdir(datadir)
        os.mkdir(os.path.join(datadir,"frames"))
        os.mkdir(os.path.join(datadir,"masks"))
        Ntot=0
        Nans=0
        for i in range(T):
            write_log(logform.format(min(i/T,0.8),0.,0.,0.))
            fr=np.array(h5[str(i)+"/frame"]).astype(np.int16)
            np.save(os.path.join(datadir,"frames","frame_"+str(i)+".npy"),fr)
            Ntot+=1

            if from_points:
                if pts_exists[i]:
                    mask=NNtools.get_mask(fr[0],pointdat[i],num_classes,grid,thres=thres,distthres=distthres).astype(np.int16)
                    np.save(os.path.join(datadir,"masks","mask_"+str(i)+".npy"),mask)
                    Nans+=1
            elif str(i)+"/mask" in h5.keys():
                np.save(os.path.join(datadir,"masks","mask_"+str(i)+".npy"),np.array(h5[str(i)+"/mask"]).astype(np.int16))
                Nans+=1
        assert Nans>0, "At least one mask is needed"



    #### Initialize the network ####
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log+="device= "+str(device)+"\n"

    NetMod = importlib.import_module(NetName)
    net=NetMod.Net(spatial_dim=3,num_in_feat=3+C,k=k_neighbor,dim_embed=dim_embed,num_classes=num_classes)
    if requeue:
        net.load_state_dict(NNtools.load_from_h5(h5[identifier+"/"+usenet]))
        log+="weight load Successful\n"
        if verbose:
            print("weight load Successful\n")
    net.to(device=device)
    n_params=sum([p.numel() for p in net.parameters()])
    log+="Total number of parameters:"+str(n_params)+"\n"

    allset=NNtools.RGNTrainDataset(datadir,shape,n=n,select_channel=0,eps=0.00001,augvec=np.array([1,1,10]),augmag=np.pi/2)
    totnum=len(allset)

    if tnum=="all" or vnum==0:#vnum should exist if tnum!="all"
        traindataloader= torch.utils.data.DataLoader(allset, batch_size=batch_size,shuffle=True, num_workers=num_workers,pin_memory=True)
        traininds=allset.indlist
        tnum=len(allset)
        vnum=0
    elif:
        if totnum==(tnum+vnum):
            tset,vset=torch.utils.data.random_split(allset,[tnum,vnum])
        else:
            tset,vset,_=torch.utils.data.random_split(allset,[tnum,vnum,totnum-tnum-vnum])
        traindataloader= torch.utils.data.DataLoader(tset, batch_size=batch_size,shuffle=True, num_workers=num_workers,pin_memory=True)
        valdataloader=torch.utils.data.DataLoader(vset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True)
        traininds=allset.indlist[tset.indices]
    else:#these should exist
        traininds
        valinds
        tindices=allset.real_ind_to_dset_ind(traininds)
        tsampler=torch.utils.data.SubsetRandomSampler(tindices)
        traindataloader= torch.utils.data.DataLoader(allset, batch_size=batch_size,sampler=tsampler, num_workers=num_workers,pin_memory=True)
        if len(valinds)!=0:
            vindices=allset.real_ind_to_dset_ind(valinds)
            vsampler=torch.utils.data.SubsetRandomSampler(vindices)
            valdataloader=torch.utils.data.DataLoader(allset,batch_size=batch_size,sampler=vsampler,num_workers=num_workers,pin_memory=True)
        tnum,vnum=len(traininds),len(valinds)


    #### requeue should match training indices
    if requeue and not adiabatic_trick:
        assert all(np.array(h5[identifier].attrs["traininds"])==traininds),"traininds not matching"

    #### save training indices
    h5[identifier].attrs["traininds"]=traininds
    log+="Training with: trainset: "+str(tnum)+" valset: "+str(vnum)+"\n"

    num_trains=len(traindataloader)
    digits=len(str(num_trains))#for pretty print
    if vnum>0:
        num_vals=len(valdataloader)
        digits_v=len(str(num_vals))

    #define the criterion function
    def selective_ce(pred_probs,labs):
        existing=torch.unique(labs)
        with torch.no_grad():
            trf=torch.zeros(num_classes).to(device=device,dtype=torch.long)
            trf[existing]=torch.arange(0,len(existing)).to(device=device,dtype=torch.long)
            labs_sel=trf[labs]
            labs_sel.requires_grad=False
        return torch.nn.CrossEntropyLoss()(pred_probs[:,existing],labs_sel)
    criterion=selective_ce

    def pts_reduce(preds_np,pts_np):
        preds_np=preds_np.reshape(-1)
        pts_np=pts_np.transpose(1,0,2).reshape(3+C,-1).T[:,:3+1]#don't see green
        pts_np[:,:3]=((pts_np[:,:3]+1)/2)*np.array([W-1,H-1,D-1])[None,:]
        pts=np.full((num_classes,3),np.nan)
        for c in range(1,num_classes):
            ind_repr=preds_np==c
            pts_repr=pts_np[:,:3][ind_repr]
            w_repr=pts_np[:,3][ind_repr]
            w=np.sum(w_repr)
            if w<((thres*eval_rep)/255):
                continue
            meanpt=np.sum(pts_repr*w_repr[:,None],axis=0)/w
            ind_repr=(preds_np==c)
            ind_repr=(ind_repr*(np.sqrt(np.sum((pts_np[:,:3]-meanpt[None,:])**2,axis=1))<distthres))
            if np.sum(ind_repr)==0:
                continue
            pts_repr=pts_np[:,:3][ind_repr]
            w_repr=pts_np[:,3][ind_repr]
            w=np.sum(w_repr)
            if w<((thres*eval_rep)/255):
                continue
            meanpt=np.sum(pts_repr*w_repr[:,None],axis=0)/w
            pts[c,:]=meanpt
        return pts

    #send to device and load the weights if requeue
    optimizer=torch.optim.Adam(net.parameters(),lr=lr)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.3,patience=patience,min_lr=5e-5)

    #copy the log if we are on requeue
    h5[identifier].attrs["log"]=log
    if requeue:
        loss=np.array(h5[identifier]["loss"])
        inds=list(loss[:,0])
        losses=list(loss[:,1])
    else:
        losses=[]
        inds=[]

    #measure time
    log+="Header Time"+str(time.time()-st)+" s \n"
    log+="Starting Training\n\n"

    if skiptrain:
        print("Skipping Training")
    elif adiabatic_trick:
        if verbose:
            print("Starting Train")
        ts=[]
        st=time.time()
        eplosses=[]
        gc=0#global count
        #typical neural network training script
        for epoch in range(num_epochs):
            ts.append(time.time()-st)
            if epoch==ep_augoff:
                text="augment is now:"+allset.change_augment("none")
                log+=text
                if verbose:
                    print(text)
            log+="Epoch: "+str(epoch)+" lr: "+str(optimizer.param_groups[0]['lr'])+"\n"
            if verbose:
                print("Epoch: "+str(epoch)+" lr: "+str(optimizer.param_groups[0]['lr']))
            net.train()
            eploss=0
            count=0
            for i,(pts,labs) in enumerate(traindataloader):
                pts = pts.to(device=device, dtype=torch.float32)
                labs= labs.to(device=device, dtype=torch.long)
                pred_probs=net(pts)
                loss=criterion(pred_probs,labs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                eploss+=loss.item()
                count+=1
                losses.append(loss.item())
                inds.append(0)

                if verbose:
                    print("    train"+str(i+1).zfill(digits)+"/"+str(num_trains)+" loss: "+str(loss.item()))
                log+="    train"+str(i+1).zfill(digits)+"/"+str(num_trains)+" loss: "+str(loss.item())+"\n"
                gc+=1
            eploss=eploss/count
            if allset.augment=="aff":
                scheduler.step(eploss)#step scheduler by epoch loss
            log+="Epoch Loss: "+str(eploss)+"\n"+"\n"
            eplosses.append(eploss)

            #save net in h5
            if "net" in h5[identifier].keys():
                del h5[identifier]["net"]
            h5.create_group(identifier+"/net")
            NNtools.save_into_h5(h5[identifier+"/net"],net.state_dict())

            #save loss
            if "loss" in h5[identifier].keys():
                del h5[identifier]["loss"]
            dset=h5[identifier].create_dataset("loss",(len(losses),2),dtype="f4",compression="gzip")
            dset[...]=np.concatenate((np.array(inds)[:,None],np.array(losses)[:,None]),axis=1).astype(np.float32)

            log+="Results saved."+"\n"+"\n"
            h5[identifier].attrs["log"]=log

            #CRITICAL, emergency break
            if os.path.exists("STOP"):
                break

        #save net in h5
        if "basenet" in h5[identifier].keys():
            del h5[identifier]["basenet"]
        h5.create_group(identifier+"/basenet")
        NNtools.save_into_h5(h5[identifier+"/basenet"],net.state_dict())
        save_backup()#make h5 backup

        #now call lineup of the frames
        if "lineup" in h5.keys():
            lineup=np.array(h5["lineup"])
        else:#if non existing, make the default lineup
            lineup=get_lineup_def()
            dset=h5.create_dataset("lineup",shape=(T,),dtype="i2")
            dset[...]=np.array(lineup).astype(np.int16)
        assert all((np.sort(np.array(traininds))==np.sort(lineup[:len(traininds)]))), "The first lineup elements should be traininds"
        save_backup()#make h5 backup

        evalset=NNtools.RGNEvalDataset(datadir,shape,n=n,select_channel=0,eps=0.00001,eval_rep=eval_rep)#we need the evalutation set for aligned indices for all frames
        lineup=list(lineup)[len(traininds):]#traininds should be at the begining
        added=[list(traininds)]#traininds are already added
        c=0
        while True:
            # regen optimizers
            optimizer=torch.optim.Adam(net.parameters(),lr=lr_adia)
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.3,patience=patience_adia,min_lr=5e-5)
            #add mask
            ichilds=[]
            breaker=False
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
                    pts,labs=evalset[ichild]
                    pts=pts.to(device=device, dtype=torch.float32)
                    pred_probs=net(pts)
                    preds_np=torch.argmax(pred_probs,dim=1).cpu().detach().numpy().astype(np.int16)
                    ptschild=pts_reduce(preds_np,pts.cpu().detach().numpy())
                    del pts,pred_probs
                #This re-masks the data
                updatemask(ichild,ptschild)
                np.save(os.path.join(datadir,"masks","mask_"+str(ichild)+".npy"),np.array(h5[identifier+"/"+str(ichild)+"/predmask"]).astype(np.int16))

            #update dataset
            allset=NNtools.RGNTrainDataset(datadir,shape,n=n,select_channel=0,eps=0.00001,augvec=np.array([1,1,10]),augmag=np.pi/2)
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
                ts.append(time.time()-st)
                # random select random memory
                t_inds_random_memory=list(np.random.choice(buff,min(num_random_memory,len(buff)),replace=False))
                # random select random memory train
                t_inds_random_memory_train=list(np.random.choice(traininds,min(num_random_memory_train,len(traininds)),replace=False))

                #all tinds
                tinds=[*t_inds_memory,*t_inds_random_memory,*t_inds_random_memory_train]
                tset=torch.utils.data.Subset(allset,allset.real_ind_to_dset_ind(tinds))
                traindataloader= torch.utils.data.DataLoader(tset, batch_size=batch_size,shuffle=True, num_workers=num_workers,pin_memory=True)
                num_trains=len(traindataloader)
                if epoch==adiabatic_ep_augoff:
                    text="augment is now:"+allset.change_augment("none")
                    log+=text
                    if verbose:
                        print(text)
                log+="Epoch: "+str(epoch)+" lr: "+str(optimizer.param_groups[0]['lr'])+"\n"
                if verbose:
                    print("Epoch: "+str(epoch)+"/"+str(ad_epochs)+" lr: "+str(optimizer.param_groups[0]['lr']))
                net.train()
                eploss=0
                count=0
                for i,(pts,labs) in enumerate(traindataloader):
                    pts= pts.to(device=device, dtype=torch.float32)
                    labs= labs.to(device=device, dtype=torch.long)
                    pred_probs=net(pts)
                    loss=criterion(pred_probs,labs)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    eploss+=loss.item()
                    count+=1
                    losses.append(loss.item())

                    inds.append(0)

                    if verbose:
                        print("    train"+str(i+1).zfill(digits)+"/"+str(num_trains)+" loss: "+str(loss.item()))
                    log+="    train"+str(i+1).zfill(digits)+"/"+str(num_trains)+" loss: "+str(loss.item())+"\n"
                    gc+=1
                eploss=eploss/count
                if allset.augment=="aff":
                    scheduler.step(eploss)
                log+="Epoch Loss: "+str(eploss)+"\n"+"\n"
                eplosses.append(eploss)

                #save net in h5
                if "net" in h5[identifier].keys():
                    del h5[identifier]["net"]
                h5.create_group(identifier+"/net")
                NNtools.save_into_h5(h5[identifier+"/net"],net.state_dict())

                #save loss
                if "loss" in h5[identifier].keys():
                    del h5[identifier]["loss"]
                dset=h5[identifier].create_dataset("loss",(len(losses),2),dtype="f4",compression="gzip")
                dset[...]=np.concatenate((np.array(inds)[:,None],np.array(losses)[:,None]),axis=1).astype(np.float32)

                log+="Results saved."+"\n"+"\n"
                h5[identifier].attrs["log"]=log

                #CRITICAL
                if os.path.exists("STOP"):
                    break

            if os.path.exists("STOP"):
                break
            c+=1
            if c%20==0:
                save_backup()#save backup, harvard rc cluster
    else:#usual neural network train
        if verbose:
            print("Starting Train")
        write_log(logform.format(1.,0.,0.,0.))
        ts=[]
        st=time.time()
        eplosses=[]
        gc=0
        for epoch in range(num_epochs):
            ts.append(time.time()-st)
            if epoch==ep_augoff:
                log+="augment is now:"+allset.change_augment("aff")
            log+="Epoch: "+str(epoch)+" lr: "+str(optimizer.param_groups[0]['lr'])+"\n"
            if verbose:
                print("Epoch: "+str(epoch)+" lr: "+str(optimizer.param_groups[0]['lr']))
            net.train()
            eploss=0
            count=0
            for i,(fr,mask) in enumerate(traindataloader):
                fr = fr.to(device=device, dtype=torch.float32)
                mask= mask.to(device=device, dtype=torch.long)
                preds=net(fr)
                loss=criterion(preds,mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                eploss+=loss.item()
                count+=1
                losses.append(loss.item())
                ious=get_ious(preds,mask,((gc%print_iou_every)!=0))
                iouss.append(ious)

                inds.append(0)

                if verbose:
                    print("    train"+str(i+1).zfill(digits)+"/"+str(num_trains)+" loss: "+str(loss.item())+" nanmeaniou: "+str(np.nanmean(ious)))
                log+="    train"+str(i+1).zfill(digits)+"/"+str(num_trains)+" loss: "+str(loss.item())+" nanmeaniou: "+str(np.nanmean(ious))+"\n"
            eploss=eploss/count
            if allset.augment=="aff":
                scheduler.step(eploss)

            log+="Epoch Loss: "+str(eploss)+"\n"+"\n"
            eplosses.append(eploss)
            if vnum>0:
                net.eval()
                log+="Validation:"+"\n"
                eploss=0
                count=0
                for i,(fr,mask) in enumerate(valdataloader):
                    fr = fr.to(device=device, dtype=torch.float32)
                    mask= mask.to(device=device, dtype=torch.long)
                    with torch.no_grad():
                        preds=net(fr)
                        loss=criterion(preds,mask)
                    losses.append(loss.item())
                    eploss+=loss.item()
                    count+=1

                    ious=get_ious(preds,mask,False)
                    iouss.append(ious)

                    inds.append(1)
                    if verbose:
                        print("    val"+str(i+1).zfill(digits_v)+"/"+str(num_vals)+" loss: "+str(loss.item())+" nanmeaniou: "+str(np.nanmean(ious)))
                    log+="    val"+str(i+1).zfill(digits_v)+"/"+str(num_vals)+" loss: "+str(loss.item())+" nanmeaniou: "+str(np.nanmean(ious))+"\n"
                eploss=eploss/count
                log+="Mean Validation Loss: "+str(eploss)+"\n"+"\n"
                eplosses.append(eploss)

            #save net in h5
            if "net" in h5[identifier].keys():
                del h5[identifier]["net"]
            h5.create_group(identifier+"/net")
            NNtools.save_into_h5(h5[identifier+"/net"],net.state_dict())

            #save loss and iou
            if "loss_iou" in h5[identifier].keys():
                del h5[identifier]["loss_iou"]
            dset=h5[identifier].create_dataset("loss_iou",(len(losses),2+len(ious)),dtype="f4",compression="gzip")
            dset[...]=np.concatenate((np.array(inds)[:,None],np.array(losses)[:,None],np.array(iouss)),axis=1).astype(np.float32)

            log+="Results saved."+"\n"+"\n"
            h5[identifier].attrs["log"]=log

            ### determine stopping
            if False and eplosses[-1]<0.01:#what condition?
                break
            #CRITICAL
            if os.path.exists("STOP"):
                break
            #log
            write_log(logform.format(1.,(epoch+1)/num_epochs,0.,0.))
        write_log(logform.format(1.,1.,0.,0.))
        ts=np.array(ts).astype(np.float32)
        if "ts" in h5[identifier].keys():
            del h5[identifier]["ts"]
        dset=h5[identifier].create_dataset("ts",ts.shape,dtype="f4")
        dset[...]=ts
        if verbose:
            print("Training Successful\n")
        log+="Training Successful\n\n"

    #save log
    h5[identifier].attrs["log"]=log

    ##Prediction phase
    evalset=NNtools.RGNEvalDataset(datadir,shape,n=n,select_channel=0,eps=0.00001,eval_rep=eval_rep)

    ptss=np.full((T,h5.attrs["N_neurons"]+1,3),np.nan)
    #predict all masks
    net.eval()
    accs=[]
    for i in range(T):
        if verbose:
            print("Evaluating "+str(i)+"/"+str(T))
        with torch.no_grad():
            pts,labs=evalset[i]
            pts=pts.to(device=device, dtype=torch.float32)
            pred_probs=net(pts)
            preds=torch.argmax(pred_probs,dim=1)
            preds_np=preds.cpu().detach().numpy().astype(np.int16)

            if labs is not None:
                labs=labs.to(device=device, dtype=torch.float32)
                num_neur_pts=torch.sum(labs!=0).item()
                acc=torch.sum((preds==labs)&(labs!=0)).item()/(num_neur_pts+1e-5)
                accs.append(acc)

        ptss[i,:num_classes]=pts_reduce(preds_np,pts.cpu().detach().numpy())
        write_log(logform.format(1.,1.,1.,i/T))


    if verbose:
        print("Prediction Complete.\n")

    log+="Prediction Complete\n\n"
    h5[identifier].attrs["log"]=log


    if "NN_pointdat" not in h5[identifier].keys():
        h5[identifier].create_dataset("NN_pointdat",ptss.shape,dtype="f4")
    h5[identifier]["NN_pointdat"][...]=ptss

    write_log(logform.format(1.,1.,1.,.1))
    log+="Point Extraction Done\n\n"


    if verbose:
        print("Run Fully Successful\n")

    log+="Run Fully Successful\n"
    h5[identifier].attrs["log"]=log

    if verbose:
        print("Repacking h5.")
    repack()

    #remove directory
    if reusedirec is None:
        shutil.rmtree(datadir)
    write_log(logform.format(1.,1.,1.,1.))

    if verbose:
        print("DONE")
except Exception as exception:
    if reusedirec is None:
        shutil.rmtree(datadir)
    raise exception
