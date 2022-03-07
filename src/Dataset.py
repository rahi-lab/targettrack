import h5py
import numpy as np
import os
supported_suffixes=["h5"]

class Dataset:
    def __init__(self,file_path):
        self.file_path=file_path
        self.suffix=self.file_path.split(".")[-1]
        assert self.suffix in supported_suffixes, suffix+" not supported"
        self.data=None

    def make(self):
        assert not os.path.exists(self.file_path),"File already present"
        if self.suffix=="h5":
            h5=h5py.File(self.file_path,"w")
            h5.close()

    def open(self):
        assert self.data is  None, "file already open"
        if self.suffix=="h5":
            self.data=h5py.File(self.file_path,"r+")

    def close(self):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            self.data.close()
            self.data=None

    def get_frame(self,time):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            return np.array(self.data[str(time)+"/frame"])

    def rename_data(self,name_before,name_after,overwrite=False):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            if name_before==name_after:
                return
            ds=self.data[name_before]
            if overwrite and name_after in self.data.keys():
                del self.data[name_after]
            dsnew=self.data.create_dataset(name_after,ds.shape,ds.dtype,ds.compression)
            dsnew=ds
            del ds

    def set_frame(self,time,frame,shape_change=False,compression="lzf"):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            key=str(time)+"/frame"
            if key not in self.data.keys():
                ds=self.data.create_dataset(key,frame.shape,frame.dtype,compression=compression)
                ds[...]=frame
            else:
                if not shape_change:
                    self.data[key][...]=frame
                else:
                    del self.data[key]
                    ds=self.data.create_dataset(key,frame.shape,frame.dtype,compression=compression)
                    ds[...]=frame

    def get_frame_z(self,time,z):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            return np.array(self.data[str(time)+"/frame"][:,:,:,z])

    def get_data_info(self):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            return dict(self.data.attrs)
    
    def update_data_info(self,dict):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            for key,val in dict.items():
                self.data.attrs[key]=val

    def get_points(self):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            if "points" not in self.data.keys():
                points=np.full((self.data.attrs["T"],self.data.attrs["N_points"]+n_add+1,3),np.nan,dtype=np.float32)
                self.set_points(points)
            return np.array(self.data["points"])

    def set_points(self,points):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            if "points" not in self.data.keys():
                ds=self.data.create_dataset("points",shape=points.shape,dtype=points.dtype)
                ds[...]=points
            else:
                self.data["points"][...]=points

    def get_helper(self,name):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            key="helper_"+name
            if key not in self.data.keys():
                print("Helper not present, bug")
                return None
            else:
                return np.array(self.data[key])

    def set_helper(self,name,helper):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            key="helper_"+name
            if key not in self.data.keys():
                ds=self.data.create_dataset(key,shape=helper.shape,dtype=helper.dtype)
                ds[...]=helper
            else:
                self.data[key][...]=helper

    def get_helper_names(self):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            ret=[]
            for key in self.data.keys():
                if key[:7]=="helper_":
                    ret.append(key[7:])
        else:
            ret=[]
        return ret

    def get_signal(self,name):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            key="signal_"+name
            if key not in self.data.keys():
                print("Signal not present, bug")
                return None
            else:
                return np.array(self.data[key])

    def get_signal_names(self):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            ret=[]
            for key in self.data.keys():
                if key[:7]=="signal_":
                    ret.append(key[7:])
        else:
            ret=[]
        return ret

    def get_data(self,name):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            key=name
            if key not in self.data.keys():
                print("Data not present, bug")
                return None
            else:
                return np.array(self.data[key])

    def get_series_names(self):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            if "series_names" in self.data.attrs.keys():
                return self.data.attrs["series_names"]
            else:
                return []
        else:
            ret=[]
        return ret

    def get_series_labels(self):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            if "series_labels" in self.data.attrs.keys():
                return self.data.attrs["series_labels"]
            else:
                return []
        else:
            ret=[]
        return ret

    def add_points(self,n_add):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            if  "points" not in self.data.keys():
                points=np.full((self.data.attrs["T"],self.data.attrs["N_points"]+n_add+1,3),np.nan,dtype=np.float32)
            else:
                points=np.array(self.data["points"])
                del self.data["points"]
                points=np.concatenate([points,np.full((points.shape[0],n_add,3),np.nan)],axis=1)

            ds=self.data.create_dataset("points",shape=points.shape,dtype=points.dtype)
            ds[...]=points
            self.data.attrs["N_points"]=self.data.attrs["N_points"]+n_add

    def set_data(self,name,data,compression="lzf",overwrite=False):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            if name in self.data.keys():
                if overwrite:
                    del self.data[name]
                else:
                    print("Cannot update",name,"with option overwrite=False")
                    return
            ds=self.data.create_dataset(name,shape=data.shape,dtype=data.dtype,compression=compression)
            ds[...]=data

    def remove(self,name):
        assert self.data is not None, "file not open"
        if self.suffix=="h5":
            if name in self.data.keys():
                del self.data[name]
