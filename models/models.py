import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import softplus, sigmoid, softmax
import pdb
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class convBlock(nn.Module):
    def __init__(self, inCh, nhid, nOp, pool=True,
                    ker=3,padding=1,pooling=2):
        super(convBlock,self).__init__()

        self.enc1 = nn.Conv2d(inCh,nhid,kernel_size=ker,padding=1)
        self.enc2 = nn.Conv2d(nhid,nOp,kernel_size=ker,padding=1)
        self.bn = nn.BatchNorm2d(inCh)  

        if pool:
            self.scale = nn.AvgPool2d(kernel_size=pooling)
        else:
            self.scale = nn.Upsample(scale_factor=pooling)
        self.pool = pool
        self.act = nn.ReLU()

    def forward(self,x):
        x = self.scale(x)
        x = self.bn(x)
        x = self.act(self.enc1(x))
        x = self.act(self.enc2(x))
        return x

class unet2d(nn.Module):
    def __init__(self, nhid=2, ker=3, inCh=1,h=128,w=128):
        super(unet2d, self).__init__()

        ### U-net Encoder with 3 downsampling layers
        self.uEnc11 = nn.Conv2d(inCh,nhid,kernel_size=ker,padding=1)
        self.uEnc12 = nn.Conv2d(nhid,nhid,kernel_size=ker,padding=1)

        self.uEnc2 = convBlock(nhid,2*nhid,2*nhid,pool=True)
        self.uEnc3 = convBlock(2*nhid,4*nhid,4*nhid,pool=True)
#        self.uEnc4 = convBlock(4*nhid,8*nhid,8*nhid,pool=True)
#        self.uEnc5 = convBlock(8*nhid,16*nhid,16*nhid,pool=True)

        ## Decoder
#        self.dec5 = convBlock(16*nhid,8*nhid,8*nhid,pool=False)
#        self.dec4 = convBlock(16*nhid,4*nhid,4*nhid,pool=False)
        self.dec3 = convBlock(4*nhid,2*nhid,2*nhid,pool=False,pooling=2)
        self.dec2 = convBlock(4*nhid,nhid,nhid,pool=False,pooling=2)

        self.dec11 = nn.Conv2d(2*nhid,nhid,kernel_size=ker,padding=1)
        self.dec12 = nn.Conv2d(nhid,1,kernel_size=ker,padding=1)
        
        self.act = nn.ReLU()

        self.h = h
        self.w = w
        self.inCh = inCh
    
    def encoder(self,x_in):
        b = x_in.shape[0]
        x_in = x_in.view(b,self.inCh,self.h,self.w)
        ### Unet Encoder
        x = []
        
        x.append(self.act(self.uEnc12(self.act(self.uEnc11(x_in)))))
        x.append(self.uEnc2(x[-1]))
        x.append(self.uEnc3(x[-1]))
#        x.append(self.uEnc4(x[-1]))
#        x.append(self.uEnc5(x[-1]))

        return x

    def decoder(self,x_enc):
        x = self.dec3(x_enc[-1])
        x = torch.cat((x,x_enc[-2]),dim=1)
        x = self.dec2(x)
        x = torch.cat((x,x_enc[-3]),dim=1)

#        x = self.dec3(x)
#        x = torch.cat((x,x_enc[-4]),dim=1)
#        x = self.dec2(x)
#        x = torch.cat((x,x_enc[-5]),dim=1)

        x = self.act(self.dec11(x))
        x = self.dec12(x)

        return x

    def forward(self, x):
        # Unet encoder result
        b = x.shape[0]
        x_enc = self.encoder(x)
        # Outputs for MSE
        xHat = torch.sigmoid(self.decoder(x_enc))

        return xHat.view(b,-1)
