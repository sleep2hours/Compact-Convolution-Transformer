"""
usage:python main.py --convnum 1 --kernelsize 3 --mids 64 --outputs 512 --encodernum 7 --h 4 --drop 0.1 --epoches 300
"""
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
from timm.loss import SoftTargetCrossEntropy
from utils import Mixup
import argparse
from CCT import CCT
import time,shutil,os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def getSummaryWriter(epochs:int, del_dir:bool):
    logdir = './logs/'
    if os.path.exists(logdir) and del_dir:
        shutil.rmtree(logdir)
    time_stamp = "{0:%Y-%m-%d/%H-%M-%S}-epoch{1}/".format(datetime.now(), epochs)
    return SummaryWriter(log_dir = logdir + time_stamp)

def label2onehot(target):
    one=torch.eye(10)
    return one[target,:]

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(10)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
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
mixup_args = {
    'mixup_alpha': 0.3,
    'cutmix_alpha': 0.,
    'cutmix_minmax': None,
    'prob': 1.1,
    'switch_prob': 0.,
    'mode': 'batch',
    'label_smoothing': 0,
    'num_classes': 10}

torch.cuda.manual_seed(1)


model=CCT(args)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
model=nn.DataParallel(model)
model.cuda()
writer=getSummaryWriter(epochs=args.epoches,del_dir=True)

def train(img,target):
    optimizer.zero_grad()
    model.train()
    img,target=img.cuda(),target.cuda()
    output=model(img)
    #loss=LabelSmoothingCrossEntropy(0.1)(output,target)
    loss=SoftTargetCrossEntropy()(output,target)
    loss.backward()
    optimizer.step()
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    accu=0
    """
        with torch.no_grad():
        accu=0
        value,index=torch.max(output,1)
        for i in range(target.size(0)):
            if target[i]==index[i]:
                accu=accu+1
    """

    return loss.data,accu

def test(img,target):
    model.eval()
    target=label2onehot(target)
    img, target = img.cuda(), target.cuda()
    with torch.no_grad():
        output = model(img)
        loss = SoftTargetCrossEntropy()(output, target)
        accu=0
        value,index=torch.max(output,1)
        for i in range(target.size(0)):
            if target[i]==index[i]:
                accu=accu+1
    return loss.data.cpu(),accu

def main():
    torch.cuda.empty_cache()
    mixup=Mixup(**mixup_args)
    num=0
    total_loss = 0
    total_accu=0
    for i in range(args.epoches):
        start_time = time.time()
        for batch_ind,(img,target) in enumerate(trainloader):
            img,target=mixup(img,target)
            loss,accu=train(img,target)
            break
            total_loss+=loss
            total_accu+=accu
            if batch_ind%24==0:
                num+=1
                writer.add_scalar('Loss/Train Loss', total_loss/24, num)
                writer.add_scalar('Acc/Train Set Accuracy', total_accu/24/64, num)
                print("%3d/%3d/%3d\tloss:%.6f\taccu:%.6f"%(i,batch_ind,len(trainloader),total_loss/24,total_accu/24/64))
                total_loss = 0
                total_accu=0
        end_time=time.time()
        print("%d-th epoch,loss:%f,time:%.3f"%(i,total_loss/len(trainloader),end_time-start_time))
        model_save_path='./model/'+str(i)+'model.tar'
        if i%5==4:
            torch.save(
            {'state_dict':model.state_dict(),'train_loss':total_loss/len(trainloader)},model_save_path
            )
        break
        test_loss=0
        test_accu=0
        for batch_ind, (img, target) in enumerate(testloader):
            if batch_ind==50:
                break
            loss,accu=test(img,target)
            test_loss+=loss
            test_accu+=accu
        writer.add_scalar('Loss/Test Loss', test_loss/50, i)
        writer.add_scalar('Acc/Test Set Accuracy', test_accu/50/64, i)
    test_loss=0
    test_accu=0
    for batch_ind, (img, target) in enumerate(testloader):
        loss,accu=test(img,target)
        test_loss+=loss
        test_accu+=accu
    writer.add_scalar('Loss/Test Loss', test_loss/len(testloader), args.epoches)
    writer.add_scalar('Acc/Test Set Accuracy', test_accu/len(testloader)/64, args.epoches)
    writer.close()
if __name__=="__main__":
    main()
