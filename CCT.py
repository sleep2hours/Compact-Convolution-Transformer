import torch
import torch.nn as nn
import torch.functional as F
import copy


def copy_layer(module,N):
    return nn.Sequential(*[copy.deepcopy(module) for i in range(N)])

class CCT(nn.Module):
    def __init__(self,args):
        super(CCT, self).__init__()
        self.conv_num=args.conv_num
        self.kernel_size=args.kernel_size
        self.inputs=args.inputs
        self.mids=args.mids
        self.outputs=args.outputs
        self.encoder_num = args.encoder_num
        self.h=args.h
        self.drop=args.drop
        self.class_num=args.class_num

        self.layer1=ConvEmbed(self.inputs,self.kernel_size,self.mids,self.outputs,self.conv_num)
        self.layer2=copy_layer(Transformer(self.h,self.outputs//self.h,self.dropout),self.encoder_num)
        self.layer3=nn.Sequential(SeqPool)
        self.layer4=nn.Linear(self.outputs,self.class_num)

        nn.init.kaiming_normal_(self.layer1)
    @staticmethod
    def init_weight(m):
        if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

    def forward(self,x):
        x=self.layer1(x)    #Embed大小:N*d_token*(HW)
        x=self.layer2(x)


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

class Encoder(nn.Module):
    def __init__(self,inputs=512,h=32,d_model=512,drop=0.1):
        super(Encoder, self).__init__()
        self.h=h
        self.d_model=d_model
        self.drop=drop
        self.inputs=inputs

        self.norm=nn.LayerNorm(d_model)
        self.layer1=MultiHead(self.h,self.d_model,self.drop)
        self.layer2=nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(self.inputs, 2 * self.inputs),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop),
            nn.Linear(2*self.inputs,inputs),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop),
        )

    def forward(self,x):
        out=x
        out=self.layer1(self.norm(out))
        x=out+x
        out=x
        x=x+self.layer2(out)
        return x

class MultiHead(nn.Module):
    def __init__(self,h,d_model,drop):
        super(MultiHead, self).__init__()
        assert d_model%h==0
        self.h=h
        self.d_model=d_model
        self.d_head=d_model//self.h




class Transformer(nn.Module):
    def __init__(self,h,d_model,dropout=0.1,encoder_num=6):
        super(Transformer, self).__init__()
        self.h=h
        self.d_model=d_model
        self.dropout=dropout




class SeqPool(nn.Module):
    def __init__(self):
        super(SeqPool, self).__init__()

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()