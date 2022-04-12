import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from timm.loss import LabelSmoothingCrossEntropy
import argparse
from CCT import CCT
import time


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

parser=argparse.ArgumentParser(description="CCT")
parser.add_argument('--convnum',type=int,default=1,help='num of embeding convolution')
parser.add_argument('--kernelsize',type=int,default=2,help='size of kernel of embeding convolution')
parser.add_argument('--mids',type=int,default=64,help='chanels of middle embeding convolution layer')
parser.add_argument('--outputs',type=int,default=512,help='token size')
parser.add_argument('--encodernum',type=int,default=4,help='num of transformer layers')
parser.add_argument('--h',type=int,default=4,help='head num of multihead')
parser.add_argument('--drop',type=float,default=0.1,help='dropout rate')
parser.add_argument('--epoches',type=int,default=10,help='dropout rate')
args=parser.parse_args()
torch.cuda.manual_seed(1)

model=CCT(args)
optimizer = optim.Adam(model.parameters(), lr=0.001)
model=nn.DataParallel(model)
model.cuda()

def train(img,target):
    model.train()
    img,target=img.cuda(),target.cuda()
    output=model(img)
    loss=LabelSmoothingCrossEntropy(0.1)(output,target)
    loss.backward()
    optimizer.step()
    return loss.data

def test(img,target):
    model.eval()
    img, target = img.cuda(), target.cuda()
    with torch.no_grad():
        output = model(img)
        loss = LabelSmoothingCrossEntropy(0.1)(output, target)
    return loss.data.cpu()

def main():
    lo=[]
    for i in range(args.epoches):
        start_time = time.time()
        total_loss=0
        for batch_ind,(img,target) in enumerate(trainloader):
            print("%d/%d"%(batch_ind,len(trainloader)))
            total_loss+=train(img,target)
        end_time=time.time()
        print("%d-th epoch,loss:%f,time:%.3f"%(i,total_loss/len(trainloader),end_time-start_time))
        lo.append(total_loss/len(trainloader))
        model_save_path='./model/'+str(i)+'model.tar'
        if i>100 and i%300==299:
            torch.save(
            {'state_dict':model.state_dict(),'train_loss':total_loss/len(trainloader)},model_save_path
            )
    lo=np.array(lo)
    axis=np.linspace(1,args.epoches,args.epoches,endpoint=True)
    plt.plot(axis,lo)
    plt.show()
    test_loss=0
    for batch_ind, (img, target) in enumerate(testloader):
        test_loss+=test(img,target)
    print("test_loss:%.5f"%(test_loss/len(testloader)))

if __name__=="__main__":
    main()


