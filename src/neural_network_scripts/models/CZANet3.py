import numpy as np
import torch
import torch.nn as nn


class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv=nn.Conv3d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.norm=nn.BatchNorm3d(out_channels)
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        return self.relu(self.norm(self.conv(x)))


class ASPP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, atrous_rates):
        super().__init__()
        self.relu=nn.ReLU(inplace=True)

        self.conv=nn.Conv3d(in_channels, mid_channels, 1, bias=False)
        self.norm=nn.BatchNorm3d(mid_channels)

        DilConvs = []
        for rate in atrous_rates:
            DilConvs.append(ASPPConv(in_channels, mid_channels, rate))

        self.DilConvs = nn.ModuleList(DilConvs)

        self.compressionconv = nn.Conv3d((len(atrous_rates)+1) * mid_channels, out_channels, 1, bias=False)
        self.compressionnorm=nn.BatchNorm3d(out_channels)

    def forward(self, x):
        res = []
        res.append(self.norm(self.conv(x)))
        for conv in self.DilConvs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.relu(self.compressionnorm(self.compressionconv(res)))


class Net(nn.Module):
    def __init__(self, n_channels=3,n_filt_init=32,growth=32,kernel_size=3,compress_targ=48,num_classes=10):

        super().__init__()

        self.relu=nn.ReLU(inplace=True)
        self.down=nn.MaxPool3d(kernel_size=2)
        self.down_noz=nn.MaxPool3d(kernel_size=(2,2,1))


        n_filt_in=n_channels
        n_filt=n_filt_init
        self.conv1=nn.Conv3d(n_filt_in, n_filt, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm1=nn.BatchNorm3d(n_filt)
        self.conv2=nn.Conv3d(n_filt, n_filt, kernel_size=5, stride=1,padding=2,bias=False)
        self.norm2=nn.BatchNorm3d(n_filt)


        n_filt_in=n_filt
        n_filt+=growth
        self.conv3=nn.Conv3d(n_filt_in, n_filt, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm3=nn.BatchNorm3d(n_filt)
        self.conv4=nn.Conv3d(n_filt, n_filt, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm4=nn.BatchNorm3d(n_filt)
        self.up1=nn.Upsample(scale_factor=2)

        #We skip one downsample in z for 2 down samples
        #two down samples are done, 64x40x8 for us
        n_filt_in=n_filt
        n_filt+=growth
        self.conv5=nn.Conv3d(n_filt_in, n_filt, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm5=nn.BatchNorm3d(n_filt)
        self.conv6=nn.Conv3d(n_filt, n_filt, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm6=nn.BatchNorm3d(n_filt)
        self.compression2conv=nn.Conv3d(n_filt, compress_targ, kernel_size=1, stride=1,bias=False)
        self.compression2norm=nn.BatchNorm3d(compress_targ)
        self.up2=nn.Upsample(scale_factor=(4,4,2))

        #still at same level 64x40x8
        n_filt_in=n_filt
        n_filt+=growth
        self.aspp2=ASPP(n_filt_in,mid_channels=n_filt,out_channels=2*n_filt, atrous_rates=((3,3,1),(6,6,2),(9,9,3)))
        self.compressionaspp2conv=nn.Conv3d(2*n_filt, compress_targ, kernel_size=1, stride=1,bias=False)
        self.compressionaspp2norm=nn.BatchNorm3d(compress_targ)#compress to send up
        #we already have up2

        #continue going down, 32x20x4 for us
        n_filt_in=n_filt
        n_filt+=growth
        self.aspp3=ASPP(n_filt_in,mid_channels=n_filt,out_channels=2*n_filt ,atrous_rates=((3,3,1),(6,6,1),(9,9,2)))
        self.compressionaspp3conv=nn.Conv3d(2*n_filt, compress_targ, kernel_size=1, stride=1,bias=False)
        self.compressionaspp3norm=nn.BatchNorm3d(compress_targ)#compress to send up
        self.up3=nn.Upsample(scale_factor=(8,8,4))

        #continue going down, 16x10x4 for us no z
        n_filt_in=n_filt
        n_filt+=growth
        self.aspp4=ASPP(n_filt_in,mid_channels=n_filt,out_channels=2*n_filt, atrous_rates=((2,2,1),(3,3,1),(4,4,2),(5,5,2)))
        self.compressionaspp4conv=nn.Conv3d(2*n_filt, compress_targ, kernel_size=1, stride=1,bias=False)
        self.compressionaspp4norm=nn.BatchNorm3d(compress_targ)#compress to send up
        self.up4=nn.Upsample(scale_factor=(16,16,4))

        self.conv_out=nn.Conv3d(n_channels+n_filt_init+(n_filt_init+growth)+4*compress_targ,num_classes, kernel_size=1, stride=1,bias=True)

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
                #print("no init",name)

    def forward(self, x,verbose=False):
        ori=x#n_channels
        if verbose:
            print(x.size())
        x=self.relu(self.norm1(self.conv1(x)))
        x=self.relu(self.norm2(self.conv2(x)))
        send0=x #n_filt_init
        x=self.down(x)

        if verbose:
            print(x.size())
        x=self.relu(self.norm3(self.conv3(x)))
        x=self.relu(self.norm4(self.conv4(x)))
        send1=self.up1(x) #n_filt_init+growth
        x=self.down_noz(x)

        #remember we don't down 64x40x8
        if verbose:
            print(x.size())
        x=self.relu(self.norm5(self.conv5(x)))
        x=self.relu(self.norm6(self.conv6(x)))#48+32
        send2=self.up2(self.relu(self.compression2norm(self.compression2conv(x)))) #64->save memory
        #compress_targ

        x=self.aspp2(x)
        send2aspp=self.up2(self.relu(self.compressionaspp2norm(self.compressionaspp2conv(x))))
        x=self.down(x)

        #remember we don't down 32x20x4
        x=self.aspp3(x)
        send3aspp=self.up3(self.relu(self.compressionaspp3norm(self.compressionaspp3conv(x))))
        x=self.down_noz(x)

        #remember we don't down 16x10x4
        x=self.aspp4(x)
        send4aspp=self.up4(self.relu(self.compressionaspp4norm(self.compressionaspp4conv(x))))

        #1+32+(32+32)+(32)+4*32
        x=torch.cat([ori,send0,send1,send2,send2aspp,send3aspp,send4aspp],dim=1)#n_channel+32+64+32+32+32

        x=self.conv_out(x)

        return x
