import torch
from torch.nn import Module
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
import math
class Conv_block(Module):
    def __init__(self,in_c,out_c,kernel=[3,3],stride=[2,2],padding=(1,1),groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding, groups=groups,bias=False)
        self.batchnorm = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.prelu(x)
        return x
'''
class mobileFaceNet(Module):
    def __init__(self):
        super(mobileFaceNet,self).__init__()
        self.conv1 = Conv_block(3,64)
        self.dw2 = Conv_block(64,64,stride=[1,1],padding=[1,1],groups=64)                       #n s
        self.bottleneck3 = BottleNeck(64,64,kernel=[3,3],stride=[2,2],padding=[1,1],groups=128) #5 2
        self.bottleneck3_1 = BottleNeck(64,64,kernel=[3,3],stride=[1,1],padding=[1,1],groups=128)
        self.bottleneck4 = BottleNeck(64,128,kernel=[3,3],stride=[2,2],padding=[1,1],groups=256)# 1 2
        self.bottleneck5 = BottleNeck(128,128,kernel=[3,3],stride=[1,1],padding=[1,1],groups=256) #6 1
        self.bottleneck6 = BottleNeck(128,128,kernel=[3,3],stride=[2,2],padding=[1,1],groups=512) #1 2
        self.bottleneck7 = BottleNeck(128,128,kernel=[3,3],stride=[1,1],padding=[1,1],groups=256) #2 1
        self.conv8 = Conv_block(128,512,kernel=[1,1],stride=[1,1],padding=[0,0])
        self.gdc_conv9 = Linear_GDC_conv(512)
        self.linear_conv10 = nn.Sequential(
            nn.Conv2d(512,128,kernel_size=[1,1],stride=[1,1],padding=[0,0],bias=False),
            nn.BatchNorm2d(128))
    def forward(self, x):
        out = self.conv1(x)  #torch.Size([32, 64, 56, 56])
        out = self.dw2(out)  #torch.Size([32, 64, 56, 56])
        out = self.bottleneck3(out) #torch.Size([32, 64, 28, 28])
        for _ in range(5-1):
            out = self.bottleneck3_1(out)  #torch.Size([32, 64, 28, 28])
        out = self.bottleneck4(out)  #torch.Size([32, 128, 14, 14])
        for _ in range(6):
            out = self.bottleneck5(out)  #torch.Size([32, 128, 14, 14])
        out = self.bottleneck6(out)   #torch.Size([32, 128, 7, 7])
        for _ in range(2):
            out = self.bottleneck7(out) #torch.Size([32, 128, 7, 7])
        out = self.conv8(out)       #torch.Size([32, 512, 7, 7])
        out = self.gdc_conv9(out)  #torch.Size([32, 512, 1, 1])
        out = self.linear_conv10(out)  #torch.Size([32, 128, 1, 1])
        return out.squeeze()   # nn.Flatten  view(batchsize,-1)
'''
Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]

class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class mobileFaceNet(nn.Module):
    def __init__(self, bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(mobileFaceNet, self).__init__()

        self.conv1 = ConvBlock(3, 64, 3, 2, 1)

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(128, 512, 1, 1, 0)

        self.linear7 = ConvBlock(512, 512, (7, 7), 1, 0, dw=True, linear=True)

        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)

        return x



class Depth_sperable_conv(Module):
    def __init__(self,in_c,out_c,kernel=[3,3],stride=[2,2],padding=(1,1),expansion=1):
        super(Depth_sperable_conv, self).__init__()
        self.depth_wise_conv = nn.Conv2d(in_c,in_c,kernel_size=kernel,stride=stride,padding=padding,groups=in_c,bias=False)
        self.batchnorm_1 = nn.BatchNorm2d(in_c)
        self.prelu = nn.PReLU(in_c)
        self.point_wise_conv = nn.Conv2d(in_c,out_c,kernel_size=[1,1],stride=[1,1],padding=0,bias=False)
        self.batchnorm_2 = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = self.depth_wise_conv(x)
        x = self.batchnorm_1(x)
        x = self.prelu(x)
        x = self.point_wise_conv(x)
        x = self.batchnorm_2(x)
        return x

class BottleNeck(Module):
    def __init__(self,in_c,out_c,kernel=[3,3],stride=[1,1],padding=(1,1),groups=1):
        super(BottleNeck,self).__init__()
        self.stride = stride
        self.conv1 = Conv_block(in_c,groups,kernel=[1,1],stride=[1,1],padding=0)
        self.d_s_conv = Depth_sperable_conv(groups,out_c,kernel=kernel,stride=stride,padding=padding)
    def forward(self,x):
        out = self.conv1(x)
        out = self.d_s_conv(out)
        if self.stride[0] == 1:
           return x+out
        else:
            return out

class Linear_GDC_conv(Module):
    def __init__(self,in_c,kernel=[7,7],stride=[1,1],padding=[0,0]):
        super(Linear_GDC_conv, self).__init__()
        self.gdc_conv = nn.Conv2d(in_c,in_c,kernel_size=kernel,stride=stride,padding=padding,groups=in_c,bias=False)
        self.bn = nn.BatchNorm2d(in_c)
    def forward(self,x):
        x = self.gdc_conv(x)
        return self.bn(x)

class Arcloss(Module):
    def __init__(self,class_num,s=32,m=0.5,easy_margin=False):
        super(Arcloss,self).__init__()
        self.class_num = class_num
        self.s= s
        self.m = m
        self.weight = Parameter(torch.Tensor(self.class_num,128))
        nn.init.xavier_normal_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi-m)
        self.mm = math.sin(math.pi -m)*m
    def forward(self,x,label):
        cosine = F.linear(F.normalize(x),F.normalize(self.weight))
        sine = torch.sqrt(1.0-torch.pow(cosine,2))
        phi = cosine*self.cos_m-sine*self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine>0,phi,cosine)
        else :
            phi = torch.where((cosine-self.th)>0,phi,cosine-self.mm)
        one_hot = torch.zeros(cosine.size(),device='cuda')
        one_hot.scatter_(1,label.view(-1,1).long(),1)
        output = (one_hot*phi) + ((1.0-one_hot)*cosine)
        output *=self.s
        return output

