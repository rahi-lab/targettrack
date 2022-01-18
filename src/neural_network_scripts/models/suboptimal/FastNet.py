import numpy as np
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, n_channels=3,growth=32,kernel_size=3,compress_targ=32,num_classes=10):

        super().__init__()

        self.relu=nn.ReLU(inplace=True)
        self.down=nn.MaxPool3d(kernel_size=2)


        n_filt_in=n_channels
        n_filt=growth
        self.conv1=nn.Conv3d(n_filt_in, n_filt, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm1=nn.BatchNorm3d(n_filt)
        self.conv2=nn.Conv3d(n_filt, n_filt, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm2=nn.BatchNorm3d(n_filt)

        n_filt_in=n_filt
        n_filt+=growth
        self.conv3=nn.Conv3d(n_filt_in, n_filt, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm3=nn.BatchNorm3d(n_filt)
        self.conv4=nn.Conv3d(n_filt, n_filt, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm4=nn.BatchNorm3d(n_filt)
        self.up1=nn.Upsample(scale_factor=2)

        n_filt_in=n_filt
        n_filt+=growth
        self.conv5=nn.Conv3d(n_filt_in, n_filt, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm5=nn.BatchNorm3d(n_filt)
        self.conv6=nn.Conv3d(n_filt, n_filt, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm6=nn.BatchNorm3d(n_filt)
        self.compression2conv=nn.Conv3d(n_filt, compress_targ, kernel_size=1, stride=1,bias=False)
        self.compression2norm=nn.BatchNorm3d(compress_targ)
        self.up2=nn.Upsample(scale_factor=4)

        n_filt_in=n_filt
        n_filt+=growth
        self.conv7=nn.Conv3d(n_filt_in, n_filt, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm7=nn.BatchNorm3d(n_filt)
        self.conv8=nn.Conv3d(n_filt, n_filt, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm8=nn.BatchNorm3d(n_filt)
        self.compression3conv=nn.Conv3d(n_filt, compress_targ, kernel_size=1, stride=1,bias=False)
        self.compression3norm=nn.BatchNorm3d(compress_targ)
        self.up3=nn.Upsample(scale_factor=8)

        n_filt_in=n_filt
        n_filt+=growth
        self.conv9=nn.Conv3d(n_filt_in, n_filt, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm9=nn.BatchNorm3d(n_filt)
        self.conv10=nn.Conv3d(n_filt, n_filt, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm10=nn.BatchNorm3d(n_filt)
        self.compression4conv=nn.Conv3d(n_filt, compress_targ, kernel_size=1, stride=1,bias=False)
        self.compression4norm=nn.BatchNorm3d(compress_targ)
        self.up4=nn.Upsample(scale_factor=16)


        self.conv_out=nn.Conv3d(n_channels+growth+2*growth+3*compress_targ,num_classes, kernel_size=1, stride=1,bias=True)


        for name, param in self.named_parameters():
            if "conv" in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)* param.size(4)
                param.data.normal_().mul_(np.sqrt(2. / n))
                #print(name)
            elif "norm" in name and 'weight' in name:
                param.data.fill_(1)
                #print(name)
            elif "norm" in name and 'bias' in name:
                param.data.fill_(0)
                #print(name)
            else:
                pass




    def forward(self, x,verbose=False):
        ori=x
        if verbose:
            print(x.size())
        x=self.relu(self.norm1(self.conv1(x)))
        x=self.relu(self.norm2(self.conv2(x)))
        send0=x #32
        x=self.down(x)

        if verbose:
            print(x.size())
        x=self.relu(self.norm3(self.conv3(x)))
        x=self.relu(self.norm4(self.conv4(x)))
        send1=self.up1(x) #64
        x=self.down(x)

        if verbose:
            print(x.size())
        x=self.relu(self.norm5(self.conv5(x)))
        x=self.relu(self.norm6(self.conv6(x)))
        send2=self.up2(self.relu(self.compression2norm(self.compression2conv(x)))) #32
        x=self.down(x)

        if verbose:
            print(x.size())
        x=self.relu(self.norm7(self.conv7(x)))
        x=self.relu(self.norm8(self.conv8(x)))
        send3=self.up3(self.relu(self.compression3norm(self.compression3conv(x)))) #32
        x=self.down(x)

        if verbose:
            print(x.size())
        x=self.relu(self.norm9(self.conv9(x)))
        x=self.relu(self.norm10(self.conv10(x)))
        send4=self.up4(self.relu(self.compression4norm(self.compression4conv(x)))) #32

        x=torch.cat([ori,send0,send1,send2,send3,send4],dim=1)#n_channel+32+64+32+32+32


        x=self.conv_out(x)

        return x
