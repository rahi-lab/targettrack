#Base code from https://github.com/gpleiss/efficient_densenet_pytorch

# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import numpy as np
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, n_channels=3,kernel_size=3,num_classes=10):

        super().__init__()

        self.relu=nn.ReLU(inplace=True)
        self.down=nn.MaxPool2d(kernel_size=2)

        # First convolution
        self.conv1=nn.Conv2d(n_channels, 32, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(32, 64, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm2=nn.BatchNorm2d(64)

        self.conv3=nn.Conv2d(64,64, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm3=nn.BatchNorm2d(64)
        self.conv4=nn.Conv2d(64,128, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm4=nn.BatchNorm2d(128)

        self.conv5=nn.Conv2d(128,128, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm5=nn.BatchNorm2d(128)
        self.conv6=nn.Conv2d(128,256, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm6=nn.BatchNorm2d(256)

        self.conv7=nn.Conv2d(256,256, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm7=nn.BatchNorm2d(256)
        self.conv8=nn.Conv2d(256,512, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm8=nn.BatchNorm2d(512)
        self.upconv1=nn.ConvTranspose2d(512,512,kernel_size=2,stride=2)

        self.conv9=nn.Conv2d(768,256, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm9=nn.BatchNorm2d(256)
        self.conv10=nn.Conv2d(256,256, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm10=nn.BatchNorm2d(256)
        self.upconv2=nn.ConvTranspose2d(256,256,kernel_size=2,stride=2)

        self.conv11=nn.Conv2d(384,128, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm11=nn.BatchNorm2d(128)
        self.conv12=nn.Conv2d(128,128, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm12=nn.BatchNorm2d(128)
        self.upconv3=nn.ConvTranspose2d(128,128,kernel_size=2,stride=2)

        self.conv13=nn.Conv2d(192,64, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm13=nn.BatchNorm2d(64)
        self.conv14=nn.Conv2d(64,64, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm14=nn.BatchNorm2d(64)
        self.conv_out=nn.Conv2d(64,num_classes, kernel_size=1, stride=1,bias=True)


        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(np.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)




    def forward(self, x,verbose=False):
        if verbose:
            print(x.size())
        x=self.relu(self.norm1(self.conv1(x)))
        x=self.relu(self.norm2(self.conv2(x)))
        if verbose:
            print(x.size())
        send1=x

        x=self.down(x)
        if verbose:
            print(x.size())
        x=self.relu(self.norm3(self.conv3(x)))
        x=self.relu(self.norm4(self.conv4(x)))
        send2=x

        x=self.down(x)
        if verbose:
            print(x.size())
        x=self.relu(self.norm5(self.conv5(x)))
        x=self.relu(self.norm6(self.conv6(x)))
        send3=x

        x=self.down(x)
        if verbose:
            print(x.size())
        x=self.relu(self.norm7(self.conv7(x)))
        x=self.relu(self.norm8(self.conv8(x)))
        x=torch.cat([send3,self.upconv1(x)],dim=1)
        if verbose:
            print(x.size())

        x=self.relu(self.norm9(self.conv9(x)))
        x=self.relu(self.norm10(self.conv10(x)))
        x=torch.cat([send2,self.upconv2(x)],dim=1)
        if verbose:
            print(x.size())

        x=self.relu(self.norm11(self.conv11(x)))
        x=self.relu(self.norm12(self.conv12(x)))
        x=torch.cat([send1,self.upconv3(x)],dim=1)
        if verbose:
            print(x.size())

        x=self.relu(self.norm13(self.conv13(x)))
        x=self.relu(self.norm14(self.conv14(x)))

        x=self.conv_out(x)

        return x
