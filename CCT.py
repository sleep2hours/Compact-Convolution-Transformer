import torch
import torch.nn as nn
import torch.functional as F
import copy


def copy_layer(module,N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CCT(nn.module):
    def __init__(self,args):
        super(CCT, self).__init__()
        self.conv_num=args.conv_num
        self.kernel_size=args.kernel_size
        self.mids=args.mids
        self.outputs=args.outputs
        self.encoder_num = args.encoder_num
        self.layer1=ConvEmbed(self.kernel_size,self.mids,self.outputs,self.conv_num)
        self.layer2=copy_layer(Transformer,self.encoder_num)
        
class ConvEmbed(nn.module):
    def __init__(self,kernel_size,mids,outputs,conv_num):
        super(ConvEmbed, self).__init__()

class Transformer(nn.module):
    def __init__(self,h,d,dropout=0.1):
        super(Transformer, self).__init__()
