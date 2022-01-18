import numpy as np
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,n_channels=3,n_z=20,out=torch.sigmoid):
        super().__init__()
        self.relu=nn.ReLU(inplace=True)
        self.down=nn.MaxPool2d(kernel_size=2)

        dim_in=n_channels
        dim_out=16
        self.conv1=nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm1=nn.BatchNorm2d(dim_out)

        dim_in=dim_out
        dim_out=32
        self.conv2=nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm2=nn.BatchNorm2d(dim_out)

        dim_in=dim_out
        dim_out=48
        self.conv3=nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm3=nn.BatchNorm2d(dim_out)

        dim_in=dim_out
        dim_out=64
        self.conv4=nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm4=nn.BatchNorm2d(dim_out)

        dim_in=dim_out
        dim_out=80
        self.conv5=nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm5=nn.BatchNorm2d(dim_out)

        dim_in=dim_out
        dim_out=96
        self.conv6=nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1,padding=1,bias=False)
        self.norm6=nn.BatchNorm2d(dim_out)

        self.lin_enc=nn.Linear(8*5*dim_out,n_z)
        self.lin_dec=nn.Linear(n_z,8*5*dim_out)

        dim_in=dim_out
        dim_out=80
        self.convt7=nn.ConvTranspose2d(dim_in, dim_out, kernel_size=2, stride=2,bias=False)
        self.conv7=nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1,bias=False)
        self.norm7=nn.BatchNorm2d(dim_out)

        dim_in=dim_out
        dim_out=64
        self.convt8=nn.ConvTranspose2d(dim_in, dim_out, kernel_size=2, stride=2,bias=False)
        self.conv8=nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1,bias=False)
        self.norm8=nn.BatchNorm2d(dim_out)

        dim_in=dim_out
        dim_out=48
        self.convt9=nn.ConvTranspose2d(dim_in, dim_out, kernel_size=2, stride=2,bias=False)
        self.conv9=nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1,bias=False)
        self.norm9=nn.BatchNorm2d(dim_out)

        dim_in=dim_out
        dim_out=32
        self.convt10=nn.ConvTranspose2d(dim_in, dim_out, kernel_size=2, stride=2,bias=False)
        self.conv10=nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1,bias=False)
        self.norm10=nn.BatchNorm2d(dim_out)

        dim_in=dim_out
        dim_out=16
        self.convt11=nn.ConvTranspose2d(dim_in, dim_out, kernel_size=2, stride=2,bias=False)
        self.conv11=nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1,bias=False)
        self.norm11=nn.BatchNorm2d(dim_out)

        self.conv_out=nn.Conv2d(dim_out,n_channels, kernel_size=3, stride=1,padding=1,bias=True)

        self.out=out

        for name, param in self.named_parameters():
            if "conv" in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
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

    def forward(self, x):
        x=self.relu(self.norm1(self.conv1(x)))
        x=self.down(x)#128,80
        x=self.relu(self.norm2(self.conv2(x)))
        x=self.down(x)#64,40
        x=self.relu(self.norm3(self.conv3(x)))
        x=self.down(x)#32,20
        x=self.relu(self.norm4(self.conv4(x)))
        x=self.down(x)#16,10
        x=self.relu(self.norm5(self.conv5(x)))
        x=self.down(x)#8,5
        x=self.relu(self.norm6(self.conv6(x)))
        x=x.reshape(x.size(0),-1)
        latent=self.lin_enc(x)
        x=self.lin_dec(latent)
        x=x.reshape(x.size(0),96,8,5)
        x=self.relu(self.norm7(self.conv7(self.convt7(x))))
        x=self.relu(self.norm8(self.conv8(self.convt8(x))))
        x=self.relu(self.norm9(self.conv9(self.convt9(x))))
        x=self.relu(self.norm10(self.conv10(self.convt10(x))))
        x=self.relu(self.norm11(self.conv11(self.convt11(x))))
        return self.out(self.conv_out(x)),latent
