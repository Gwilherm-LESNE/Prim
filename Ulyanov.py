# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time 
import PIL.Image
import torchvision.models as models
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

print('## Imports: done ##')

#%%     
      
def preprocess_img(x): #x is a 3D-numpy array 
    x = resize(x,(3,224,224))
    temp = torch.tensor(x,dtype=torch.float)
    out = torch.unsqueeze(temp,0)
    return out

def get_activation(activation,name):
    def hook(model, input, output):
            activation[name] = output.detach()
    return hook

def gram_matrix(x):
    result = torch.einsum('bcij,bdij->bcd', x, x)
    shape = x.shape
    temp = shape[2]*shape[3]
    return result/temp

def texture_loss(output,target,src_vgg19,ipt_vgg19):
    trans=transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor()])
    output=torch.unsqueeze(trans(output),0)
    target=torch.unsqueeze(trans(target),0)
        
    source_activation = {}
    input_activation = {}
    index_list=[2,9,16,29,42]
    name_list=['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
    for i,index in enumerate(index_list):    
        src_vgg19[index].register_forward_hook(get_activation(source_activation,name_list[i]))
        ipt_vgg19[index].register_forward_hook(get_activation(input_activation,name_list[i]))
    
    ipt_vgg19(output)
    src_vgg19(target)
        
    loss=torch.tensor(0.0)  
    for idx,name in enumerate(name_list):
        temp = torch.pow(gram_matrix(input_activation[name])-gram_matrix(source_activation[name]),2)
        loss = loss + torch.sum(temp)
    return loss

def texture_loss_bis(output,target):
    trans=transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor()])
    output=torch.unsqueeze(trans(output),0)
    target=torch.unsqueeze(trans(target),0)
    
    loss=torch.tensor(0.0)
    index_list=[2,9,16,29,42]
    activNet = ActivNet(index_list)
    activNet.cuda()
    activNet.eval()
    for idx,_ in enumerate(index_list):
        temp = torch.pow(gram_matrix(activNet.get_activ(output,idx))-gram_matrix(activNet.get_activ(target,idx)),2)
        loss = loss + torch.sum(temp)
    return loss

def custom_loss(outputs,targets,src_vgg,ipt_vgg):#outputs & targets are 4D arrays, first dimension is batch size
    batch_loss = torch.tensor(0.0)
    for i in range(outputs.shape[0]):
        batch_loss = batch_loss + texture_loss(outputs[i],targets[i],src_vgg,ipt_vgg)
    return batch_loss

def custom_loss_bis(outputs,targets):#outputs & targets are 4D arrays, first dimension is batch size
    batch_loss = torch.tensor(0.0)
    for i in range(outputs.shape[0]):
        batch_loss = batch_loss + texture_loss_bis(outputs[i],targets[i])
    return batch_loss

class ActivNet(nn.Module):
    def __init__(self,index_list=[2,9,16,29,42]):
        super(ActivNet, self).__init__()
        self.features0 = nn.Sequential(*list(vgg19.features.children())[:index_list[0]+1])
        self.features1 = nn.Sequential(*list(vgg19.features.children())[:index_list[1]+1])
        self.features2 = nn.Sequential(*list(vgg19.features.children())[:index_list[2]+1])
        self.features3 = nn.Sequential(*list(vgg19.features.children())[:index_list[3]+1])
        self.features4 = nn.Sequential(*list(vgg19.features.children())[:index_list[4]+1])
        
    def get_activ(self,x,idx):
        if (idx==0):
            return self.features0(x)
        elif (idx==1):
            return self.features1(x)
        elif (idx==2):
            return self.features2(x)
        elif (idx==3):
            return self.features3(x)
        elif (idx==4):
            return self.features4(x)
        
    def eval(self):
        self.features0.eval()
        self.features1.eval()
        self.features2.eval()
        self.features3.eval()
        self.features4.eval()
    
    def cuda(self):
      self.features0.cuda()
      self.features1.cuda()
      self.features2.cuda()
      self.features3.cuda()
      self.features4.cuda()

class TextureNet(nn.Module):

    def __init__(self):
        super(TextureNet, self).__init__()

        self.block1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,8,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8,8,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8,8,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(8))
        
        self.block2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,8,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8,8,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8,8,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(8))
        
        self.block3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,8,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8,8,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8,8,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(8))
        
        self.block4_1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,8,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8,8,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8,8,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(8))
        
        self.block5_1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,8,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8,8,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8,8,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(scale_factor=2,mode='nearest'),
            torch.nn.BatchNorm2d(8))
        
        self.block4_2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,16,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16,16,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16,16,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(scale_factor=2,mode='nearest'),
            torch.nn.BatchNorm2d(16))
        
        self.block3_2 = torch.nn.Sequential(
            torch.nn.Conv2d(24,24,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(24),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(24,24,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(24),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(24,24,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(24),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(scale_factor=2,mode='nearest'),
            torch.nn.BatchNorm2d(24))
        
        self.block2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32,32,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(scale_factor=2,mode='nearest'),
            torch.nn.BatchNorm2d(32))
        
        self.block1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(40,40,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(40),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(40,40,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(40),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(40,40,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(40),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(40,3,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(3),
            torch.nn.LeakyReLU())
        
    def forward(self,x_0,x_1,x_2,x_3,x_4):
        
        x5 = self.block5_1(x_4)
        x4 = self.block4_1(x_3)
        x3 = self.block3_1(x_2)
        x2 = self.block2_1(x_1)
        x1 = self.block1_1(x_0)
        
        x4 = torch.cat((x5,x4),dim=1)
        x4 = self.block4_2(x4)
        
        x3 = torch.cat((x4,x3),dim=1)
        x3 = self.block3_2(x3)
        
        x2 = torch.cat((x3,x2),dim=1)
        x2 = self.block2_2(x2)
        
        x1 = torch.cat((x2,x1),dim=1)
        out = self.block1_2(x1)
        
        return out
    
    def cuda(self):
        self.block1_1.cuda()
        self.block2_1.cuda()
        self.block3_1.cuda()
        self.block4_1.cuda()
        self.block5_1.cuda()
        self.block1_2.cuda()
        self.block2_2.cuda()
        self.block3_2.cuda()
        self.block4_2.cuda()
    
class NoiseTextureDataset(Dataset):
    
    def __init__(self, size, texture_path):
        self.size = size
        
        img = plt.imread('texture7.jpg')/255.    
        img = np.array(img, dtype=np.float32)
        text = torch.tensor(np.transpose(img,(2,0,1)))
        self.texture = torch.unsqueeze(text,0)
        
        noise0 = np.random.randn(size,1,3,256,256)/9+0.5
        noise1 = np.random.randn(size,1,3,128,128)/9+0.5
        noise2 = np.random.randn(size,1,3,64,64)/9+0.5
        noise3 = np.random.randn(size,1,3,32,32)/9+0.5
        noise4 = np.random.randn(size,1,3,16,16)/9+0.5
            
        self.noise0 = torch.tensor(noise0,dtype=torch.float)
        self.noise1 = torch.tensor(noise1,dtype=torch.float)
        self.noise2 = torch.tensor(noise2,dtype=torch.float)
        self.noise3 = torch.tensor(noise3,dtype=torch.float)
        self.noise4 = torch.tensor(noise4,dtype=torch.float)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()       
        sample = {'noise0': self.noise0[idx],
                  'noise1': self.noise1[idx],
                  'noise2': self.noise2[idx],
                  'noise3': self.noise3[idx],
                  'noise4': self.noise4[idx],
                  'texture': self.texture}     
        return sample

def epoch(data, model, optimizer,train_history, device, src_vgg=None, ipt_vgg=None):
    start_time = time.time()
    total_train_loss = 0
    count=0
    for i, sample in enumerate(data):
        
        noise0 = sample['noise0'].to(device)
        noise1 = sample['noise1'].to(device)
        noise2 = sample['noise2'].to(device)
        noise3 = sample['noise3'].to(device)
        noise4 = sample['noise4'].to(device)
        targets = sample['texture'].to(device)

        # forward + backward
        outputs = model(noise0,noise1,noise2,noise3,noise4)
        #loss = custom_loss(outputs, targets, src_vgg, ipt_vgg)
        loss = custom_loss_bis(outputs,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        count+=1
   
    print("train_loss: {:.2f} took: {:.2f}s".format(total_train_loss / count,time.time() - start_time))
    train_history.append(total_train_loss / count)

def train(net, batch_size, n_epochs, learning_rate, texture_path):
    
    print("====== HYPERPARAMETERS ======")
    print("batch_size=", batch_size)
    print("n_epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 29)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #src_vgg19 = models.vgg19_bn(pretrained=True)
    #src_net = src_vgg19.features
    #src_net.eval()
    #for p in src_net.parameters():
    #    p.requires_grad = False
    #ipt_vgg19 = models.vgg19_bn(pretrained=True)
    #ipt_net = ipt_vgg19.features
    #ipt_net.eval()
    #for p in ipt_net.parameters():
    #    p.requires_grad = False
    
    model = net
    model = model.to(device)
    #model.cuda()
    optimizer = torch.optim.Adam(model.parameters(recurse=True),learning_rate)
    lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    for p in model.parameters():
      p.requires_grad=True
    
    dataset = NoiseTextureDataset(batch_size, texture_path)
    #train = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
    train_history = []

    for i in range(n_epochs):
        print("=================\n==== EPOCH "+str(i+1)+" ====\n=================\n")
        epoch(dataset, model, optimizer, train_history, device)
        lr_sched.step()
    return train_history

print('## Defining functions: done ##')  
#%%

#text7 = plt.imread('texture7.jpg')/255
#x0 = torch.tensor(np.random.randn(1,3,224,224)/9 + 0.5,dtype=torch.float)
#
#Download VGG19 models for loss computing
#src_vgg19 = models.vgg19_bn(pretrained=True)
#src_vgg19.eval()
#ipt_vgg19 = models.vgg19_bn(pretrained=True)
#ipt_vgg19.eval()
#
#print(texture_loss(x0,preprocess_img(text7),src_vgg19,ipt_vgg19)) #Test the loss function

print('## Setting variables: done ##')
#%%
myNet=TextureNet()
vgg19 = models.vgg19_bn(pretrained=True)   
history = train(myNet, 16, 10, 0.01, 'texture7.jpg')


