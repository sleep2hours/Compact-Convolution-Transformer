import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

def copy_layer(module,N):
    return nn.Sequential(*[copy.deepcopy(module) for i in range(N)])

def copy_layer_list(module,N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CCT(nn.Module):
    def __init__(self,args):
        super(CCT, self).__init__()
        self.conv_num=args.convnum
        self.kernel_size=args.kernelsize
        self.inputs=3
        self.mids=args.mids
        self.outputs=args.outputs      #Embeding的长度d_model
        self.encoder_num = args.encodernum
        self.h=args.h
        self.drop=args.drop

        self.embed=ConvEmbed(self.inputs,self.kernel_size,self.mids,self.outputs,self.conv_num)
        self.transformer=copy_layer(Encoder(self.h,self.outputs,self.drop),self.encoder_num)
        self.seqpool=nn.Sequential(SeqPool(self.outputs))
        self.MLP=nn.Linear(self.outputs,self.outputs*2)
        self.classify=nn.Linear(self.outputs*2,10)

        self.apply(CCT.init_weight)

    @staticmethod
    def init_weight(m):
        if isinstance(m,nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

    def forward(self,x):
        x=self.embed(x)    #Embed大小:N*d_token*(HW)
        x=self.transformer(x)
        x=self.seqpool(x)
        x=self.MLP(x)
        return self.classify(x).squeeze(1)


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
        self.layer2=None
        if conv_num>2:
            self.layer2=copy_layer(ConvEmbed.make_convs(mids,mids,k_size=kernel_size,p_size=3,p_stride=1,p_pad=1),conv_num-2)
        self.layer3=ConvEmbed.make_convs(mids,outputs,k_size=kernel_size,p_size=3,p_stride=2,p_pad=1)

    def forward(self,x):
        x=self.layer1(x)
        if self.layer2 is not None:
            x=self.layer2(x)
        x=self.layer3(x)    #x的size:N*d_token*H*W
        x=x.contiguous()
        N,d,H,W=x.size()
        return x.view(N,d,H*W).transpose(1,2).contiguous()   #N*(H/2W/2)*d

class Encoder(nn.Module):
    def __init__(self,h=32,d_model=512,drop=0.1):
        super(Encoder, self).__init__()
        self.h=h
        self.d_model=d_model
        self.drop=drop

        self.norm=nn.LayerNorm(d_model)
        self.layer1=MultiHead(self.h,self.d_model,self.drop)
        self.layer2=nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(self.d_model, 2 * self.d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop),
            nn.Linear(2*self.d_model,d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop),
        )

    def forward(self,x):
        x=self.norm(x)
        out=self.layer1(x,x,x)+x
        x=out
        x=x+self.layer2(out)
        return x

class MultiHead(nn.Module):
    def __init__(self,h=8,d_model=512,drop=0.1):
        super(MultiHead, self).__init__()
        assert d_model%h==0
        self.h=h
        self.d_model=d_model
        self.d_head=d_model//self.h
        self.drop=nn.Dropout(drop)
        self.norm = copy_layer_list(nn.LayerNorm(d_model),3)
        self.linearhead=copy_layer_list(nn.Linear(self.d_model,self.d_model),4)

    @staticmethod
    def make_mask(size):
        mask=np.triu(np.ones((size,size)),k=1).astype('uint8')
        mask=torch.from_numpy(mask)==0
        return mask.cuda()

    @staticmethod
    def attention(query,key,value,mask,dropout):
        d_k=query.size(-1)
        score=torch.matmul(query,key.transpose(-1,-2))/d_k**0.5
        if mask is not None:

            score=score.masked_fill(mask==0,-1e9)
        atten=F.softmax(score,-1)
        if dropout is not None:
            atten=dropout(atten)
        return torch.matmul(atten,value)

    def forward(self,query,key,value):
        N_batch=query.size(0)
        query,key,value=[n(x) for n,x in zip(self.norm,(query,key,value))]
        query,key,value=[l(x).view(N_batch,-1,self.h,self.d_head).transpose(1,2) for l,x in zip(self.linearhead,(query,key,value))]   #q,k,v:N*h*length*d_head
        mask=MultiHead.make_mask(query.size(2))
        x=MultiHead.attention(query,key,value,mask,self.drop).transpose(1,2).contiguous().view(N_batch,-1,self.d_model)
        return self.linearhead[-1](x)

class SeqPool(nn.Module):
    def __init__(self,d_model):
        super(SeqPool, self).__init__()
        self.d_model=d_model

        self.linear=nn.Linear(self.d_model,1)
    def forward(self,x):
        x_l=self.linear(x).transpose(1,2).contiguous()
        x_l=F.softmax(x_l,-1)
        return torch.matmul(x_l,x)




