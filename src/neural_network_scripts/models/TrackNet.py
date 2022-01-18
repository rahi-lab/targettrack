#Base code from https://github.com/gpleiss/efficient_densenet_pytorch

# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features,growth_rate,kernel_size):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,growth_rate=growth_rate,kernel_size=kernel_size)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        for name, layer in self.named_children():
            x = torch.cat([x,layer(x)],dim=1)
        return x


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate,kernel_size):
        super().__init__()
        self.conv=nn.Conv3d(num_input_features, growth_rate,kernel_size=kernel_size, stride=1, padding=2)
        self.relu=nn.ReLU(inplace=True)
        self.norm=nn.BatchNorm3d(growth_rate)

    def forward(self, x):
        x=self.relu(self.norm(self.conv(x)))
        return x

class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.poolconv=nn.Conv3d(num_input_features,num_output_features,kernel_size=4, stride=2,padding=1)
        self.norm=nn.BatchNorm3d(num_output_features)
    def forward(self,x):
        return self.norm(self.poolconv(x))

class _TransitionUp(nn.Module):
    def __init__(self,num_features_in,num_features_out,convupsplit=[0.5,0.5]):
        super().__init__()
        self.numconvout=int(num_features_out*convupsplit[0]/np.sum(convupsplit))
        self.numupsampout=num_features_out-self.numconvout
        self.uppoolconv=nn.ConvTranspose3d(num_features_in, self.numconvout,kernel_size=2,stride=2)
        self.upsamp=nn.Upsample(scale_factor=2)
        self.upsampcompre=nn.Conv3d(num_features_in,self.numupsampout,kernel_size=1)
        self.norm=nn.BatchNorm3d(self.numconvout+self.numupsampout)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        x_convup=self.uppoolconv(x)
        x_upsamp=self.upsampcompre(self.upsamp(x))
        x=torch.cat([x_convup,x_upsamp],dim=1)
        x=self.relu(self.norm(x))
        return x


class Net(nn.Module):
    def __init__(self, n_channels=3,growth_rate=16, block_config=(3,3,3,3),bottleneck_num_layers=3, compression=1.0,num_init_features=12,kernel_size=5,num_classes=10):

        super().__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'

        # First convolution
        self.conv3d0=nn.Conv3d(n_channels, num_init_features, kernel_size=3, stride=1,padding=1)
        self.norm3d0=nn.BatchNorm3d(num_init_features)
        self.conv3d1=nn.Conv3d(num_init_features, num_init_features, kernel_size=3, stride=1,padding=1)
        self.norm3d1=nn.BatchNorm3d(num_init_features)
        self.relu=nn.ReLU(inplace=True)
 


        # Each denseblock
        self.dense_downs=[]
        self.trans_downs=[]
        num_features_list=[n_channels+num_init_features]
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):

            block = _DenseBlock(num_layers=num_layers,num_input_features=num_features,growth_rate=growth_rate,kernel_size=kernel_size)
            self.add_module('denseblock%d' % (i + 1),block)
            self.dense_downs.append(block)
            num_features = num_features + num_layers * growth_rate
            num_features_list.append(num_features)

            trans = _Transition(num_input_features=num_features,num_output_features=int(num_features*compression))
            self.add_module('transition%d' % (i + 1),trans)
            self.trans_downs.append(trans)
            num_features = int(num_features * compression)

        block = _DenseBlock(num_layers=bottleneck_num_layers,num_input_features=num_features,growth_rate=growth_rate,kernel_size=kernel_size)
        self.add_module('bottleneck',block)
        num_features_list.append(bottleneck_num_layers*growth_rate)

        num_features_list=num_features_list[::-1]

        #upsampling
        self.dense_ups=[]
        self.trans_ups=[]
        self.num_keep_filters_list=[num_features_list[0]]
        #for bottleneck up
        for i, num_layers in enumerate(list(block_config)[::-1]):
            trans =_TransitionUp(self.num_keep_filters_list[-1],self.num_keep_filters_list[-1])
            self.add_module('transitionup%d' % (i + 1),trans)
            self.trans_ups.append(trans)

            num_input_features=self.num_keep_filters_list[-1]+num_features_list[i+1]
            block = _DenseBlock(num_layers=num_layers,num_input_features=num_input_features,growth_rate=growth_rate,kernel_size=kernel_size)
            self.add_module('denseblockup%d' % (i + 1),block)
            self.dense_ups.append(block)

            num_keep_filters=growth_rate*num_layers
            self.num_keep_filters_list.append(num_keep_filters)

        self.num_final_kept_filters=num_input_features+self.num_keep_filters_list[-1]+num_features_list[-1]#keep the result and original resolution maps
        self.add_module('conv_out',nn.Conv3d(self.num_final_kept_filters,num_classes, kernel_size=1, stride=1,bias=True))


        for name, param in self.named_parameters():
            #print(name)
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(np.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)




    def forward(self, x,verbose=False):
        ori=x
        saves=[]
        #conv3d
        x=self.relu(self.norm3d0(self.conv3d0(x)))
        x=self.relu(self.norm3d1(self.conv3d1(x)))
        #conv2d
        saves.append(x)
        if verbose:
            print("init features",x.size(),np.prod(x.size()))
            print("start down")
        for dense_down,trans_down in zip(self.dense_downs,self.trans_downs):
            x=dense_down(x)
            if verbose:
                print("  dense",x.size(),np.prod(x.size()))
            saves.append(x)
            x=trans_down(x)
            if verbose:
                print("  trans",x.size(),np.prod(x.size()))
        if verbose:
            print("start bottleneck")
        x=self.bottleneck(x)
        if verbose:
            print("  bottleneck",x.size(),np.prod(x.size()))
            print("start up")
        for i,(dense_up,trans_up,num_keep_filters) in enumerate(zip(self.dense_ups,self.trans_ups,self.num_keep_filters_list)):
            if verbose:
                print("  sending up",x[:,-num_keep_filters:].size(),np.prod(x[:,-num_keep_filters:].size()))
            x=trans_up(x[:,-num_keep_filters:])
            x=torch.cat([x,saves[-(i+1)]],1)
            if verbose:
                print("  cat with skip",saves[-(i+1)].size())
            x=dense_up(x)
            saves[-(i+1)]=None
            if verbose:
                print("  dense",x.size(),np.prod(x.size()))
        i+=1
        if verbose:
                print("cat with skip and ori",saves[-(i+1)].size(),ori.size())
        x=torch.cat([x,saves[-(i+1)],ori],1)

        if verbose:
            print(x.size(),np.prod(x.size()))

        x=self.conv_out(x)#this is all
        if verbose:
            print(x.size(),np.prod(x.size()))


        return x
        