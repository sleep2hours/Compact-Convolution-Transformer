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
        self.encoder_num=args.encoder_num
        
        self.layer1=copy_layer(ConvEmbed,self.conv_num)
        
class ConvEmbed(nn.module):
    def __init__(self):
        super(ConvEmbed, self).__init__()
