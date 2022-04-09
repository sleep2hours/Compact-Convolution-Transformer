import torch
import torch.nn as nn
import torch.functional as F
import copy
from torch.autograd import Variable

def copy_layer(module,N):
    return nn.Sequential(*[copy.deepcopy(module) for i in range(N)])

class ConvEmbed(nn.Module):
    @staticmethod
    def make_convs(inputs,outputs,k_size=3,p_size=3,p_stride=2,p_pad=1):
        return nn.Sequential(nn.Conv2d(inputs,outputs,kernel_size=k_size,stride=1,padding=k_size//2),
                             nn.ReLU(inplace=True),
                             nn.MaxPool2d(kernel_size=p_size,stride=p_stride,padding=p_pad)
                             )
    def __init__(self,inputs=3,kernel_size=3,mids=64,outputs=512,conv_num=7):  #输入通道数、卷积核的大小、中间层卷积核的个数、最终输出的通道数（token的维度数）、卷积层的个数
        super(ConvEmbed, self).__init__()
        self.layer1=ConvEmbed.make_convs(inputs,mids,k_size=kernel_size,p_size=3,p_stride=1,p_pad=1)
        self.layer2=copy_layer(ConvEmbed.make_convs(mids,mids,k_size=kernel_size,p_size=3,p_stride=1,p_pad=1),conv_num-2)
        self.layer3=ConvEmbed.make_convs(mids,outputs,k_size=kernel_size,p_size=3,p_stride=2,p_pad=1)

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)    #x的size:N*d_token*H*W
        x=x.contiguous()
        N,d,H,W=x.size()
        return x.view(N,d,H*W)

t=ConvEmbed()
x=torch.arange(4*4*3).view(1,3,4,4).float()
x=nn.Dropout(0.5)(x)
m = nn.Dropout(p=0.5)
a=Variable(torch.tensor([5.,4.]),requires_grad=True)
b=Variable(torch.tensor([2.,3.]))
y=(a**2+5*b)
z=y.mean()
z.backward()
print(a.grad)
print(b.grad)
