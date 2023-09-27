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
import scipy.spatial as spat
from scipy.ndimage import affine_transform
import copy
from skimage.color import rgb2gray
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
import scipy.ndimage as sim

import targeted_augmentation_objects3

torch.autograd.set_detect_anomaly(True)
#### This is the grammar to parse the command line parameters
sys.path.append("src/neural_network_scripts/models")
st=time.time()
#assert len(sys.argv)==3
## Don't change
logfn=sys.argv[2]#MB :the address of the log file
dataset_path=sys.argv[1]# a copy of the whole data set is in this path
dataset_name=os.path.split(dataset_path)[1].split(".")[0]#MB first split pathname into a pair.component [1] has no slash
props=dataset_name.split("_")
NetName=props[-2]
print("NetName:  "+NetName)
runname=props[-1]
print("runname:  "+runname)
identifier="net/"+NetName+"_"+runname
GetTrain = int(sys.argv[5])#train on previous training set or not
print("GetTrain:"+str(GetTrain))
#### These are the run options
########################################################################
#Run status
deformInput=int(sys.argv[3])#determines whether to generate the deformed frames or not
if deformInput==1:
    deformation_trick=True
else:
    deformation_trick=False
adiabatic_trick=False
verbose=True
if deformInput==2:
    requeue=True
    skiptrain=True
else:
    requeue=False
    skiptrain=False
if requeue:
    usenet="net"



#Run detail
channel_num=2


##directory handling
reusedirec="data/data_temp/zmdir"


#neural network training parameters
batch_size=1
#determines what augmentation method is used and when that is stopped and replaced by affine transformations
aug_dict={0:"aff_cut",50:"aff"}
num_epochs=int(sys.argv[4])#number of epochs

ep_augoff=30
lr=0.003
patience=8
num_workers=8
print_iou_every=10


if deformation_trick:
    num_additional=int(sys.argv[8])#number of deformed frames you are generating
    deformation_num_epochs=2
    deformation_augdict={0:"aff_cut"}

########################################################################

####Check the dependencies
dependencies=["W","H","D","C","T","N_neurons"]
h5=h5py.File(dataset_path,"r+")
for dep in dependencies:
    if dep not in h5.attrs.keys():
        h5.close()
        assert False, "Dependency "+dep+" not in  attributes"
T=h5.attrs["T"]
if "oldT" in h5.attrs.keys():
    origfrNum = h5.attrs["oldT"]
else:
    origfrNum = h5.attrs["T"]
C,W,H,D=h5.attrs["C"],h5.attrs["W"],h5.attrs["H"],h5.attrs["D"]#x,y,z ordering
if deformation_trick:
    if not "distmat" in h5.keys():
        distmat=NNtools.Compute_distmat(h5,T,W,H,batch_size=20,n_z=31,n_channels=1)
channel_num=min(channel_num,C)
shape=(channel_num,W,H,D)#x,y,z ordering



#### logging file function, this is the runtime log
def write_log(txt,end="\n"):
    with open(logfn,"w") as f:
        f.write(txt+end)
logform="Prepare={:04.4f} Train={:04.4f} Predict={:04.4f} GetPoints={:04.4f}"
write_log(logform.format(0.,0.,0.,0.))

#### saving backup function
def repack():# MB: closes current h5 and opens a new one
    global h5
    h5.close()
    NNtools.repack(dataset_path)
    h5=h5py.File(dataset_path,"r+")
def save_backup():#MB: closes, saves and opens the file in readable format again
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

if reusedirec is not None:#MB: use the dir of masks and frames for training if it already exists
    datadir=reusedirec
else:
    datadir=os.path.join("data","data_temp",dataset_name)


#### The computation loop is all in a try except to delete the data
if verbose:
    print("Preparing...")
try:
    num_classes=h5.attrs["N_neurons"]+1


    ####Unpack for fast, multiprocessed loading, unless already unpacked
    #MB: if not already done, this part saves all the frames and all the masks (which are less than the number of frames.)

    DeforemeFrames = int(sys.argv[6])#whether or not add the deformed frames?
    if reusedirec is None or not os.path.exists(reusedirec):   # TODO: is this correct?
        os.mkdir(datadir)
        os.mkdir(os.path.join(datadir,"frames"))
        os.mkdir(os.path.join(datadir,"highs"))
        os.mkdir(os.path.join(datadir,"masks"))
        Ntot=0
        Nans=0
        print("unpacking frames")#MB check
        for i in range(T):# I think this unpacks all the segmented/mask frames -MB
            write_log(logform.format(min(i/T,0.8),0.,0.,0.))
            fr=np.array(h5[str(i)+"/frame"]).astype(np.int16)   # TODO: access to dataset with methods?
            np.save(os.path.join(datadir,"frames","frame_"+str(i)+".npy"),fr)
            Ntot+=1

            if str(i)+"/high" in h5.keys():
                np.save(os.path.join(datadir,"highs","high_"+str(i)+".npy"),np.array(h5[str(i)+"/high"]).astype(np.int16))
            else:
                np.save(os.path.join(datadir,"highs","high_"+str(i)+".npy"),np.full((1,W,H),255).astype(np.int16))

            if GetTrain == 0:
                if str(i)+"/mask" in h5.keys():
                    np.save(os.path.join(datadir,"masks","mask_"+str(i)+".npy"),np.array(h5[str(i)+"/mask"]).astype(np.int16))
                    Nans+=1
        if GetTrain == 1:
            k = 0#index of the dataset you want to copy, usually 0 for the first datasset
            NNname = list(h5['net'].keys())#the new network is NNname[0] so use NNname[1] to access previous networks training set
            traininds = h5['net'][NNname[k]].attrs['traininds']#train indices are saved as an attribute of the first dataset(run w/O deformation)
            for j in traininds:#placing the training frames and masks in the deformation folder
                np.save(os.path.join(datadir,"masks","mask_"+str(j)+".npy"),np.array(h5[str(j)+"/mask"]).astype(np.int16))
                Nans+=1
            if DeforemeFrames == 1:
                num_added_frames = int(sys.argv[7])
                if num_added_frames==0:
                    for l in range(origfrNum,T):#to add the deformed frames
                        np.save(os.path.join(datadir,"masks","mask_"+str(l)+".npy"),np.array(h5[str(l)+"/mask"]).astype(np.int16))
                        Nans+=1
                else:
                    for l in range(origfrNum,origfrNum+num_added_frames):#to add the deformed frames
                        np.save(os.path.join(datadir,"masks","mask_"+str(l)+".npy"),np.array(h5[str(l)+"/mask"]).astype(np.int16))
                        Nans+=1

            tnum="all"
            vnum=0

        assert Nans>0, "At least one mask is needed"

    allset=NNtools.TrainDataset(datadir,shape)

    ''' MB added the following section to get the right total number of cells as categories'''
    U =set() #MB added
    for i in allset.indlist :
        U=U.union(set(np.unique(h5[str(i)+"/mask"])))
    num_classes = len(U)#MB added

    #### Initialize the network ####
    if (W+H)>600:
        device = torch.device('cpu')#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log+="device= "+str(device)+"\n"
    NetMod = importlib.import_module(NetName)
    net=NetMod.Net(n_channels=shape[0],num_classes=num_classes)
    if requeue:
        net.load_state_dict(NNtools.load_from_h5(h5[identifier+"/"+usenet])) #MB: dictionary of h5[identifier] items where the values are converted to torch tensors.
        log+="weight load Successful\n"
        if verbose:
            print("weight load Successful\n")
    net.to(device=device)
    n_params=sum([p.numel() for p in net.parameters()])
    log+="Total number of parameters:"+str(n_params)+"\n"

    allset=NNtools.TrainDataset(datadir,shape)
    ''' MB: added the following section to get the right total number of cells as categories'''
    print("Total number of cells in the segmented frames")
    print(num_classes)
    print(U)
    totnum=len(allset)
    print("number of annotated frames: ")
    print(totnum)

    if True:#GetTrain == 0 :
        '''
        partition the dataset from scratch to training set and validation set
        '''
        if GetTrain == 0:
            tnum = int(sys.argv[7])
            vnum = int(sys.argv[8])
        if deformInput==2:
            tnum=1
            vnum=1    
        if tnum=="all" or vnum==0:
            traindataloader= torch.utils.data.DataLoader(allset, batch_size=batch_size,shuffle=True, num_workers=num_workers,pin_memory=True)
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
        h5[identifier].attrs["traininds"]=traininds
        if GetTrain == 0:
            if vnum>0:
                h5[identifier].attrs["Validinds"]=allset.indlist[vset.indices]# MB added
            else:
                h5[identifier].attrs["Validinds"]=[]

    Ut =set() #MB added
    for i in traininds :
        Ut=Ut.union(set(np.unique(h5[str(i)+"/mask"])))
    current_classes = len(Ut)#MB added
    print("existing classes in the training set are:")
    print(Ut)

    #### requeue should match training indices
    if requeue and not adiabatic_trick:
        assert all(np.array(h5[identifier].attrs["traininds"])==traininds),"traininds not matching"

    #### save training indices

    log+="Training with: trainset: "+str(tnum)+" valset: "+str(vnum)+"\n"

    num_trains=len(traindataloader)
    digits=len(str(num_trains))#for pretty print
    if vnum>0:
        num_vals=len(valdataloader)
        digits_v=len(str(num_vals))

    #define the iou function (intersection over union)
    '''This is a measure of how good the object detection task is performed.
    iou = (the intersection of area of neuron i in ground truth and predicted mask)/
     (union of i's areain GT and predicted mask)
    '''
    def get_ious(preds,mask,skip=False):
        if skip:
            return np.full(num_classes,np.nan)
        maskgot=torch.argmax(preds,dim=1)
        ioubins=np.zeros(num_classes)
        for i in range(num_classes):
            thismask=(mask==i)#MB: area of the neuron i in ground truth(GT) mask
            if torch.sum(thismask).item()==0:#MB: the case when neuron i is absent in GT
                ioubins[i]=np.nan
                continue
            thismaskgot=(maskgot==i)#MB: area of the neuron i in predicted mask
            intersection=torch.sum(thismask&thismaskgot).item()
            union=torch.sum(thismask|thismaskgot).item()
            ioubins[i]=intersection/union
        return ioubins

    #define the criterion function
    def selective_ce(pred_raw,target_mask):
        existing=torch.unique(target_mask)
        with torch.no_grad():
            trf=torch.zeros(num_classes).to(device=device,dtype=torch.long)
            trf[existing]=torch.arange(0,len(existing)).to(device=device,dtype=torch.long)
            mask=trf[target_mask]
            mask.requires_grad=False
        return torch.nn.CrossEntropyLoss()(pred_raw[:,existing],mask)
    criterion=selective_ce

    #send to device and load the weights if requeue
    optimizer=torch.optim.Adam(net.parameters(),lr=lr)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.3,patience=patience,min_lr=5e-5)

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

    #measure time
    log+="Header Time"+str(time.time()-st)+" s \n"
    log+="Starting Training\n\n"
    #if int(sys.argv[3])==1:
    #    skiptrain=True
    if skiptrain:
        print("Skipping Training")
    else:# not defTrick:#usual neural network train
        if verbose:
            print("Starting Train")
        ts=[]
        st=time.time()
        eplosses=[]
        gc=0#global count
        #typical neural network training script
        if not int(sys.argv[3]):
            for epoch in range(num_epochs):
                ts.append(time.time()-st)
                if epoch in aug_dict.keys():
                    text="augment is now:"+allset.change_augment(aug_dict[epoch])
                    log+=text
                    if verbose:
                        print(text)
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
                        print("    train"+str(i+1).zfill(digits)+"/"+str(num_trains)+" loss: "+str(loss.item())+" nanmeaniou: "+str("nan" if np.all(np.isnan(ious)) else np.nanmean(ious)))
                    log+="    train"+str(i+1).zfill(digits)+"/"+str(num_trains)+" loss: "+str(loss.item())+" nanmeaniou: "+str("nan" if np.all(np.isnan(ious)) else np.nanmean(ious))+"\n"
                    gc+=1
                eploss=eploss/count
                if allset.augment=="aff":
                    scheduler.step(eploss)#step scheduler by epoch loss
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

                #CRITICAL, emergency break
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
            '''
            MB:
            adding the new deformation technique for augmentation
            '''

        if deformation_trick:
            import cv2
            def noisy(noise_typ,image):
                if noise_typ == "gauss":
                    row,col,ch= image.shape
                    mean = 0
                    var = 0.001
                    sigma = var**0.005
                    gauss = np.random.normal(mean,sigma,(row,col,ch))
                    gauss = gauss.reshape(row,col,ch)
                    gauss = np.maximum(0,gauss)
                    noisy = image + gauss
                    return noisy
                elif noise_typ == "s&p":
                    row,col,ch = image.shape
                    s_vs_p = 0.5
                    amount = 0.004
                    out = np.copy(image)
                    # Salt mode
                    num_salt = np.ceil(amount * image.size * s_vs_p)
                    coords = [np.random.randint(0, i - 1, int(num_salt))
                            for i in image.shape]
                    out[coords] = 1

                    # Pepper mode
                    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
                    coords = [np.random.randint(0, i - 1, int(num_pepper))
                            for i in image.shape]
                    out[coords] = 0
                    return out
                elif noise_typ == "poisson":
                    vals = len(np.unique(image))
                    vals = 2 ** np.ceil(np.log2(vals))
                    noisy = np.random.poisson(image * vals) / float(vals)
                    return noisy
                elif noise_typ =="speckle":
                    row,col,ch = image.shape
                    gauss = np.random.randn(row,col,ch)
                    gauss = gauss.reshape(row,col,ch)
                    noisy = image + image * gauss
                    return noisy
            defTrick = int(sys.argv[7])#whether to use the def trick designed for the point data or the masks
            if verbose:
                print("Adding Deformed Frames")
            #copy the existing masks first
            dir_deformations=os.path.join(datadir,"deformations")
            if os.path.exists(dir_deformations):
                shutil.rmtree(dir_deformations)#remove the deformed frames from the previous runs
            os.mkdir(dir_deformations)
            os.mkdir(os.path.join(datadir,"deformations","frames"))
            os.mkdir(os.path.join(datadir,"deformations","masks"))
            for name in np.array(allset.filelist)[allset.real_ind_to_dset_ind(traininds)]:#placing the training frames and masks in the deformation folder
                i=int(os.path.split(name)[-1].split(".")[0].split("_")[1])
                shutil.copyfile(os.path.join(datadir,"frames","frame_"+str(i)+".npy"),os.path.join(datadir,"deformations","frames","frame_"+str(i)+".npy"))
                shutil.copyfile(name,os.path.join(datadir,"deformations","masks","mask_"+str(i)+".npy"))

            if defTrick == 3:
                deformMethod =  int(sys.argv[9])
                with torch.no_grad():
                    plots_dir = os.path.join(datadir, 'targeted_augmentation_plots')
                    if not os.path.exists(plots_dir):
                        os.makedirs(plots_dir)
                    plot_results = True
                    targeted_augmentation_objects3.targeted_augmentation(h5, num_additional, datadir, allset, traininds,
                                                                        T, identifier, shape,num_classes, plot_results=plot_results,
                                                                        plots_dir=plots_dir,method = deformMethod)
            else:
                distmat=np.array(h5["distmat"])
                additional_inds=NNtools.select_additional(T,traininds,distmat,num_additional)[len(traininds):]#index of frames used for augmentation
                h5[identifier].attrs["Deforminds"]=additional_inds# MB added
                evalset=NNtools.EvalDataset(datadir,shape)#gets the new frame(out side of training set) .npy files
                countpl = 0
                totFrameChecked = 0
                ExtframeCount = 0
                ManualCheck = 1#whether to ask the user interactively about the acceptance of the frames  or not
                ignoreNN = 1#ignore the NN that has jusn been run
                if ignoreNN:
                    NNname = list(h5['net'].keys())
                dum = 0
                while num_additional>0:
                    totFrameChecked = totFrameChecked + num_additional
                    num_additional = 0

                    for i in additional_inds:
                        with torch.no_grad():
                            fr,mask=evalset[i]#new target frame
                            fr_i=fr.unsqueeze(0).to(device=device, dtype=torch.float32)
                            if ignoreNN ==1:
                                knn="net/"+NNname[0]+"/"+str(i)+"/predmask"
                                pred = h5[knn]#prediction of the target frame
                                predmask = torch.tensor(pred)
                                predmask = predmask.cpu().detach().numpy().astype(np.int16)#prediction of the frame out side of training set

                            else:
                                pred=net(fr_i)#prediction of the target frame
                                predmask=torch.argmax(pred[0],dim=0).cpu().detach().numpy().astype(np.int16)#prediction of the frame out side of training set

                        if len(np.unique(predmask))<(num_classes)-2:#choose masks that  arebetter
                            num_additional = num_additional+1
                            print("frame not accepted")
                        else:
                            with torch.no_grad():
                                i_parent=traininds[np.argmin(distmat[traininds,i])]#closest training set frame to frame i
                                fr,mask=evalset[i_parent]
                                maskTemp = mask.cpu().detach().numpy().astype(np.int16)
                                pts_parent, pts_child,ptaug = NNtools.get_pts_from_masksJV(maskTemp,predmask)#pts_child: the training set points that we are transforming.point_parent:training set points after tranformation. ptaug: goal of the transformation which is the predicted mask
                                pts_parent=pts_parent.astype(np.float32)
                                NanElements = np.argwhere(np.isnan(pts_child[:,1]))
                                pts_parent = np.delete(pts_parent,NanElements,0)
                                pts_child = np.delete(pts_child,NanElements,0)


                                fr_r = fr_i[0,0].cpu().detach().numpy()
                                if countpl < 100 and ManualCheck==1:
                                    import matplotlib.pyplot as plt
                                    fig, axs = plt.subplots(1, 3)
                                    axisfontSize=13
                                    axs[0].set_title('starting frame ',fontsize=axisfontSize)
                                    axs[1].set_title('deformed frame',fontsize=axisfontSize)
                                    axs[2].set_title('target frame',fontsize=axisfontSize)
                                    fig2, axs2 = plt.subplots(4,2)
                                    im00=axs2[0,0].imshow(np.max(fr_r,axis=2).T)#plot the target frame
                                    fig.colorbar(im00, ax=axs2[0, 0])
                                    axs2[0,0].set_ylabel('target:'+str(i),rotation=0, labelpad=26)
                                    axs2[0,0].scatter(ptaug[:,0],ptaug[:,1],c="w",s=1)
                                    axs[2].imshow(np.max(fr_r,axis=2).T)#plot the target frame
                                    im01 = axs2[0,1].imshow(np.max(predmask,axis=2).T)#,cmap="nipy_spectral",interpolation="none")
                                    fig.colorbar(im01, ax=axs2[0, 1])
                                    axs2[0,1].set_ylabel('target mask',rotation=0, labelpad=26)
                            transform_temp, loss_affine = NNtools.rotation_translation(pts_child,pts_parent)#find a linear transformation+translation that takes the two sets of points to each other
                            rot = transform_temp[:,:3]
                            offset = transform_temp[:,3]

                            fr,mask=evalset[i_parent]
                            maskTemp = mask.cpu().detach().numpy().astype(np.int16)
                            #fr_temp = np.clip(fr[0].cpu().detach().numpy()*255,0,255).astype(np.int16)#NewChange
                            fr_temp = fr[0].cpu().detach().numpy()
                            order = 3   # default value of affine_transform
                            cval = np.median(fr_temp)  # why median??
                            mode='constant'
                            frRot = affine_transform(fr_temp,rot,offset,mode=mode,cval = cval, order= order)#change new to fr

                            order = 0
                            cval = 0
                            maskTempRot = affine_transform(maskTemp,rot,offset,mode=mode,cval = cval, order=order)#change new to fr

                            frRot_clip = frRot
                            if countpl < 100 and ManualCheck==1:
                                im30 = axs2[3,0].imshow(np.max(frRot_clip,axis=2).T)
                                fig.colorbar(im30, ax=axs2[3, 0])
                                axs2[3,0].set_ylabel("Rotated frame",rotation=0, labelpad=26)


                            if defTrick==1:
                                pts_Rot, pts_Rotts,ps_pred = NNtools.get_pts_from_masksJV(maskTempRot,predmask)
                                deformation,_,_,success=NNtools.get_deformation(ptfrom=pts_Rot,ptto=pts_Rotts,sh=(W,H,D),k_cut_dimless=2,iterations=400,lambda_div=1,scale=(1,1,1),device=device)
                                if success!=0:
                                    print("Deformation failed due to ",success)
                                    continue
                            if defTrick==2:
                                mask_warp = maskTempRot
                                x, y, z = np.nonzero(mask_warp)
                                '''
                                #find center of mass for each mask object
                                Cells = np.unique(maskTempRot)
                                Cells = np.sort(Cells)
                                Vol = np.zeros(len(Cells))
                                CoM = np.zeros([len(Cells),3])
                                for i in range(len(Cells)):
                                    Vol[i] = np.sum(maskTempRot==Cells[i])
                                    Coor = np.nonzero(maskTempRot==Cells[i])
                                    CoM[i] = [int(np.mean(Coor[0])), int(np.mean(Coor[1])), int(np.mean(Coor[2]))]
                                '''
                                zmin =  np.min(z)-1
                                z_start =np.maximum(0,zmin)
                                sh = np.shape(predmask)
                                z_end = np.minimum(sh[2],np.max(z)+1)
                                image1_warp = frRot

                                MaxPix = 20
                                mask_warp = mask_warp/MaxPix
                                mask_warp2 = mask_warp
                                for l in range(z_start,z_end):
                                    # --- Compute the optical flow
                                    v, u = optical_flow_tvl1(fr_r[:,:,l], frRot[:,:,l])
                                    nr, nc = predmask[:,:,0].shape

                                    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
                                    mask_warp2[:,:,l] = warp(mask_warp[:,:,l], np.array([row_coords + v, col_coords + u]), mode='nearest')
                                    image1_warp[:,:,l] = warp(frRot[:,:,l], np.array([row_coords + v, col_coords + u]), mode='nearest')
                                mask_warp2 = mask_warp2*MaxPix
                                diag=0
                                #setting mode of correction
                                if diag==1:
                                    s = sim.generate_binary_structure(3,3)#considered together even if they touch diagonally
                                    labelArray, numFtr = sim.label(mask_warp2>0, structure=s)
                                else:
                                    labelArray, numFtr = sim.label(mask_warp2>0)
                                print(numFtr)
                                for j in range(numFtr+1):
                                    submask =  (labelArray==j)
                                    #print(submask[1,2,3])
                                    #for k in range(len(Cells)):
                                    #    print(k)
                                    #    print(CoM[k,0])
                                    #    if submask[int(CoM[k,0]),int(CoM[k,1]),int(CoM[k,2])]:
                                    #            mask_warp2[submask]=Cells[i]
                                    '''
                                    origmask =  (maskTempRot>0)
                                    list = np.unique(maskTempRot[submask&origmask])
                                    print(list)
                                    if len(list)>0:
                                        mask_warp2[submask]=list[0]#np.max(np.unique(mask_warp2[submask]))
                                    else:
                                    '''

                                    origmask =  (maskTempRot>0)
                                    list = np.unique(maskTempRot[submask&origmask])
                                    if len(list)==2:
                                        submaskOrig1 =  (maskTempRot==list[0])
                                        submaskOrig2 =  (maskTempRot==list[1])
                                        mask_warp2[submask&submaskOrig1]=list[0]#np.max(np.unique(mask_warp2[submask]))
                                        mask_warp2[submask&submaskOrig2]=list[1]
                                        if len(set(mask_warp2[submask])-set(list)) > 0:
                                            coorX,coorY,coorZ = np.nonzero(np.array(submask))
                                            for C in range(len(coorX)):
                                                if not mask_warp2[coorX[C],coorY[C],coorZ[C]] == list[0]:
                                                    if not mask_warp2[coorX[C],coorY[C],coorZ[C]] == list[1]:
                                                        if list[0] in np.unique(mask_warp2[coorX[C]-1:coorX[C]+1,coorY[C],coorZ[C]]) or list[0] in np.unique(mask_warp2[coorX[C],coorY[C]-1:coorY[C]+1,coorZ[C]]) or list[0] in np.unique(mask_warp2[coorX[C],coorY[C],coorZ[C]-1:coorZ[C]+1]):
                                                            mask_warp2[coorX[C],coorY[C],coorZ[C]] = 100#list[0]
                                                        elif list[1] in np.unique(mask_warp2[coorX[C]-1:coorX[C]+1,coorY[C],coorZ[C]]) or list[1] in np.unique(mask_warp2[coorX[C],coorY[C]-1:coorY[C]+1,coorZ[C]]) or list[1] in np.unique(mask_warp2[coorX[C],coorY[C],coorZ[C]-1:coorZ[C]+1]):
                                                            mask_warp2[coorX[C],coorY[C],coorZ[C]] = 101#list[1]
                                                        else:
                                                            mask_warp2[coorX[C],coorY[C],coorZ[C]] = 0
                                            mask_warp2[mask_warp2==100] = list[0]
                                            mask_warp2[mask_warp2==101] = list[1]
                                    else:
                                        mask_warp2[submask]=np.max(np.unique(mask_warp2[submask]))

                            if countpl < 100 and ManualCheck==1:
                                im31 = axs2[3,1].imshow(np.max(maskTempRot,axis=2).T)
                                fig.colorbar(im31, ax=axs2[3, 1])
                                fr_rParent=fr.unsqueeze(0).to(device=device, dtype=torch.float32)#added for plotting
                                fr_rParent = fr_rParent[0,0].cpu().detach().numpy()


                                im10 = axs2[1,0].imshow(np.max(fr_rParent,axis=2).T)
                                axs2[1,0].set_ylabel('train frame:'+str(i_parent),rotation=0, labelpad=26)
                                fig.colorbar(im10, ax=axs2[1, 0])
                                axs[0].imshow(np.max(fr_rParent,axis=2).T)

                                im11 = axs2[1,1].imshow(np.max(mask.cpu().detach().numpy(),axis=2).T)#,cmap="nipy_spectral",interpolation="none")
                                fig.colorbar(im11, ax=axs2[1, 1])
                                axs2[1,1].scatter(pts_parent[:,0],pts_parent[:,1],c="w",s=1)
                                axs2[1,1].scatter(pts_child[:,0],pts_child[:,1],c="b",s=1)

                            if defTrick==1:

                                frA,maskDum=evalset[i_parent]
                                fr2 = torch.Tensor(frRot)
                                frA[0]= fr2
                                fr3=frA.unsqueeze(0).to(device=device, dtype=torch.float32)

                                mask = maskTempRot
                                mask = torch.Tensor(mask)
                                mask=mask.unsqueeze(0).to(device=device, dtype=torch.float32)
                                frB,mask=NNtools.deform((W,H,D),deformation,fr3,mask=mask)#apply the deformation on the training frame and its mask
                                frC=np.clip(frB[0].cpu().detach().numpy()*255,0,255).astype(np.int16)

                                mask=mask[0].cpu().detach().numpy().astype(np.int16)


                                if countpl < 100 and ManualCheck==1:
                                    plt.subplot(4,2,5)
                                    plt.imshow(np.max(frC[0],axis=2).T)
                                    plt.set_ylabel(str(distmat[i_parent,i]),rotation=0, labelpad=26)
                                    plt.colorbar()
                                    plt.subplot(4,2,6)
                                    plt.imshow(np.max(mask,axis=2).T)#,cmap="nipy_spectral",interpolation="none")
                                    plt.scatter(pts_child[:,0],pts_child[:,1],c="blue",s=1)
                                    plt.set_ylabel(str(distmat[i_parent,i]),rotation=0, labelpad=26)
                                    plt.subplot(4,2,8)
                                    plt.imshow(np.max(maskTempRot,axis=2).T)#,cmap="nipy_spectral",interpolation="none")
                                    plt.scatter(pts_Rot[:,0],pts_Rot[:,1],c="blue",s=1)
                                    plt.show()
                                    countpl = countpl + 1

                            if defTrick == 2:
                                if countpl < 100 and ManualCheck==1:

                                    im20=axs2[2,0].imshow(np.max(image1_warp,axis=2).T)
                                    axs2[2,0].set_ylabel('warped image',rotation=0, labelpad=26)
                                    fig.colorbar(im20, ax=axs2[2, 0])
                                    axs[1].imshow(np.max(image1_warp,axis=2).T)
                                    fig.subplots_adjust(wspace=0.270)

                                    im21 = axs2[2,1].imshow(np.max(mask_warp2,axis=2).T)#,cmap="nipy_spectral",interpolation="none")
                                    axs2[2,1].set_ylabel('warped mask',rotation=0, labelpad=26)
                                    fig.colorbar(im21, ax=axs2[2, 1])

                                    fig.savefig(str(i)+'.pdf')
                                    fig2.savefig(str(i)+'_2.pdf')


                                countpl = countpl + 1
                                mask = mask_warp2
                                ship = np.shape(image1_warp)
                                frC = np.zeros([1,ship[0],ship[1],ship[2]])
                                addNoise=0
                                if addNoise==1:
                                    image1_warp = noisy("poisson",image1_warp)
                                frC[0]= np.clip(image1_warp*255,0,255)

                                checkCode = 0 #MB check
                                if checkCode==1:
                                    sfr = traininds[dum]#SegFrames[dum]
                                    if True:# sfr not in traininds:
                                        fr,mask=evalset[sfr]
                                        mask = mask.cpu().detach().numpy().astype(np.int16)
                                        frC = np.clip(fr.cpu().detach().numpy()*255,0,255).astype(np.int16)#NewChange
                                    else:
                                        print(sfr)
                                    dum = dum+1


                            if ManualCheck==1 :
                                checkMB = 1#int(input("Do you accept this frame?(press 1 for yes) "))#12
                            else:
                                checkMB = 1
                            if checkMB==1:
                                h5.attrs["oldT"]=T
                                np.save(os.path.join(datadir,"deformations","frames","frame_"+str(T+ExtframeCount)+".npy"),frC) #we add T to avoid collision, but don't want to define another class
                                np.save(os.path.join(datadir,"deformations","masks","mask_"+str(T+ExtframeCount)+".npy"),mask)
                                dset=h5.create_dataset(str(T+ExtframeCount)+"/frame",fr.shape,  dtype="i2", compression="gzip")#to save the mask in data set
                                dset[...] = frC
                                dset=h5.create_dataset(str(T+ExtframeCount)+"/mask",mask.shape,  dtype="i2", compression="gzip")
                                dset[...] = mask
                                ExtframeCount = ExtframeCount+1
                                h5.attrs["T"] = T+ExtframeCount
                            else:
                                num_additional = num_additional+1
                    if num_additional > 0 :# if not enough frames were added
                        additional_inds=NNtools.select_additional(T,traininds,distmat,totFrameChecked + num_additional)[len(traininds)+totFrameChecked:]
                #now add the new masks
                ContinueNNWDef = 0#int(input("Do you like to continue this NN training?(press 1 for yes) "))#13
                if ContinueNNWDef == 1 :
                    allset=NNtools.TrainDataset(os.path.join(datadir,"deformations"),shape,high=False)
                    traindataloader= torch.utils.data.DataLoader(allset, batch_size=batch_size,shuffle=True, num_workers=num_workers,pin_memory=True)
                    valdataloader=torch.utils.data.DataLoader(vset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True)
                    optimizer=torch.optim.Adam(net.parameters(),lr=lr)
                    NNtools.trainMask(h5,identifier,device,net,optimizer,criterion,\
                        NNtools.get_ious,None,allset,traindataloader,deformation_num_epochs,deformation_augdict,\
                        log,verbose,losses,iouss,inds,print_iou_every,digits,len(traindataloader),vnum=vnum,\
                        valdataloader=valdataloader,digits_v=len(valdataloader),num_vals=vnum,write_log=True,\
                        logfn=logfn)



    #save log
    h5[identifier].attrs["log"]=log
    if not int(sys.argv[3])==1:
        print("predictions running:")
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
                    ious=get_ious(pred,mask,False)
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

        if verbose:
            print("Prediction Complete.\n")

        log+="Prediction Complete\n\n"
        h5[identifier].attrs["log"]=log

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
