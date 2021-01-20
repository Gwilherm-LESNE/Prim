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
from torch.autograd import Function
import copy
import scipy.fft

print('## Imports: done ##')
      
#%%

def preprocess_img(x): #x is a 3D-numpy array 
    x = resize(x,(3,256,256))
    temp = torch.tensor(x,dtype=torch.float)
    out = torch.unsqueeze(temp,0)
    return out

def gram_matrix(x):
    result = torch.einsum('cij,dij->cd', x[0], x[0])
    shape = x.shape
    temp = shape[1]*shape[2]
    return result/temp

def texture_loss_bis(output,target):
    loss=torch.tensor(0.0)
    #index_list=[2,9,16,29,42]
    index_list=[2]
    activNet = ActivNet(index_list=[2,9,16,29,42])
    activNet.cuda()
    activNet.eval()
    preprocess = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    for idx,_ in enumerate(index_list):
        temp = torch.pow(gram_matrix(activNet.get_activ(torch.unsqueeze(preprocess(output),dim=0),idx))-gram_matrix(activNet.get_activ(torch.unsqueeze(preprocess(target),dim=0),idx)),2)
        loss = loss + torch.sum(temp)
    return loss

def custom_loss_bis(outputs,targets):#outputs & targets are 4D arrays, first dimension is batch size
    batch_loss = torch.tensor(0.0)
    for i in range(outputs.shape[0]):
        batch_loss = batch_loss + texture_loss_bis(outputs[i],targets[i])
    return batch_loss

def spectrum_dist(src_img,ipt_img): #inputs are 4d tensors of shape (1,c,w,h)
    src = src_img[0].type(torch.complex128).permute((1,2,0))
    ipt = ipt_img[0].type(torch.complex128).permute((1,2,0))
    fi = torch.fft.fftn(src)
    fi_hat = torch.fft.fftn(ipt)
    prod = torch.mul(fi_hat,torch.conj(fi))
    norm = torch.div(prod,torch.absolute(prod))
    i_tilde = torch.fft.ifftn(torch.mul(norm,fi))
    output = torch.dist(torch.absolute(ipt), torch.absolute(i_tilde), 2)
    return (1/2) * output**2

def texture_loss(output,target,spectrum=False): #Used for Gatys  
    loss=torch.tensor(0.0)
    index_list=[2,9,16,29,42]
    activNet = ActivNet(index_list)
    activNet.cuda()
    activNet.eval()
    for idx,_ in enumerate(index_list):
        temp = torch.pow(gram_matrix(output[idx])-gram_matrix(activNet.get_activ(target,idx)),2)
        loss = loss + torch.sum(temp)
    if spectrum:
        loss += 1e5 * spectrum_dist(target,output[-1])
    return loss

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
    
    def forward(self,x0):
        return [self.features0(x0),self.features1(x0),self.features2(x0),self.features3(x0),self.features4(x0),x0]

class TextureNet(nn.Module):

    def __init__(self):
        super(TextureNet, self).__init__()

        self.block1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,8,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8,8,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8,8,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(8))
        
        self.block1_1.apply(weights_init)
        
        self.block2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,8,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8,8,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8,8,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(8))
        
        self.block2_1.apply(weights_init)

        self.block3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,8,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8,8,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8,8,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(8))
        
        self.block3_1.apply(weights_init)

        self.block4_1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,8,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8,8,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8,8,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(8))
        
        self.block4_1.apply(weights_init)

        self.block5_1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,8,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8,8,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8,8,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(scale_factor=2,mode='nearest'),
            torch.nn.BatchNorm2d(8))
        
        self.block5_1.apply(weights_init)

        self.block4_2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,16,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16,16,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16,16,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(scale_factor=2,mode='nearest'),
            torch.nn.BatchNorm2d(16))
        
        self.block4_2.apply(weights_init)

        self.block3_2 = torch.nn.Sequential(
            torch.nn.Conv2d(24,24,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(24),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(24,24,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(24),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(24,24,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(24),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(scale_factor=2,mode='nearest'),
            torch.nn.BatchNorm2d(24))
        
        self.block3_2.apply(weights_init)

        self.block2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32,32,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(scale_factor=2,mode='nearest'),
            torch.nn.BatchNorm2d(32))
        
        self.block2_2.apply(weights_init)

        self.block1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(40,40,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(40),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(40,40,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(40),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(40,40,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(40),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(40,3,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(3),
            torch.nn.LeakyReLU())
        
        self.block1_2.apply(weights_init)

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
      self.block5_1.cuda()
      self.block4_1.cuda()
      self.block3_1.cuda()
      self.block2_1.cuda()
      self.block1_1.cuda()
      self.block4_2.cuda()
      self.block3_2.cuda()
      self.block2_2.cuda()
      self.block1_2.cuda()
    
class NoiseTextureDataset(Dataset):
    
    def __init__(self, size, texture_path):
        self.size = size
        
        if texture_path == 'noise':
          img = torch.rand((256,256,3),dtype=torch.float)
        else:
          img = plt.imread(texture_path)/255.
          img = np.array(img, dtype=np.float32)

        text = torch.tensor(np.transpose(img,(2,0,1)))
        #preprocess = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        #text = preprocess(text)
        self.texture = torch.unsqueeze(text,0)
            
        self.noise0 = torch.rand((size,1,3,256,256),dtype=torch.float)
        self.noise1 = torch.rand((size,1,3,128,128),dtype=torch.float)
        self.noise2 = torch.rand((size,1,3,64,64),dtype=torch.float)
        self.noise3 = torch.rand((size,1,3,32,32),dtype=torch.float)
        self.noise4 = torch.rand((size,1,3,16,16),dtype=torch.float)

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

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

def epoch(data, model, optimizer,train_history, device, n_iter):
    start_time = time.time()
    total_train_loss = 0.
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
        optimizer.zero_grad()
        #outputs = torch.clamp(outputs,0.,1.)
        loss = custom_loss_bis(outputs, targets)
        loss.backward()
        for p in model.parameters(recurse=True):
            if (p.grad == None):
                print(p)
            p.grad /= torch.norm(p.grad, p=2, dim=0)
        optimizer.step()
        '''
        #update learning rate
        if (n_iter%500 == 1):
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 0.8
        '''
        total_train_loss += loss
        '''
        #use it if using LBFGS Loss
        def closure():
          outputs = model(noise0,noise1,noise2,noise3,noise4)
          optimizer.zero_grad()
          #outputs = torch.clamp(outputs,0.,1.)
          loss = custom_loss_bis(outputs, targets)
          loss.backward()
          return loss

        optimizer.step(closure)
        '''
        count+=1
   
    print("train_loss: {:.5f} took: {:.2f}s".format(total_train_loss / count,time.time() - start_time))
    train_history.append(total_train_loss / count)

def train(net, batch_size, n_epochs, learning_rate, texture_path):
    
    print("====== HYPERPARAMETERS ======")
    print("batch_size=", batch_size)
    print("n_epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 29)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = net
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(recurse=True),lr=learning_rate)
    #optimizer = torch.optim.LBFGS(model.parameters())
    lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for p in model.parameters():
      p.requires_grad=True
    
    dataset = NoiseTextureDataset(batch_size, texture_path)
    train_history = []

    # On itère sur les epochs
    for i in range(n_epochs):
        print("\n==== EPOCH "+str(i+1)+" ====\n")
        epoch(dataset, model, optimizer, train_history, device, i)
        #To use if you want to update the learning rate
        if (i%10 == 1) and (i>10):
            lr_sched.step()
    return train_history

def gatys(text,ite=20,spectrum=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.rand((1,3,224,224),dtype=torch.float)
    x0=x0.to(device)
    x0.requires_grad = True
    x0 = x0.cuda()
    text=text.to(device)

    optimizer = torch.optim.LBFGS([x0])
    #optimizer = torch.optim.Adam([x0],lr=1)
    activNet = ActivNet([2,9,16,29,42])
    activNet.cuda()
    activNet.eval()

    new_img = x0.clone().cpu()[0]
    plt.figure()
    plt.imshow(np.transpose(new_img.detach().numpy(),(1,2,0)))

    for i in range(ite):
      def closure():
          optimizer.zero_grad()
          out = activNet(x0)
          loss = texture_loss(out,text,spectrum)
          loss.backward()
          return loss
      print(".",end="")
      optimizer.step(closure)
    return x0  

#generator's convolutional blocks 2D
class Conv_block2D(nn.Module):
    def __init__(self, n_ch_in, n_ch_out, m=0.1):
        super(Conv_block2D, self).__init__()

        self.conv1 = nn.Conv2d(n_ch_in, n_ch_out, 3, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(n_ch_out, momentum=m)
        self.conv2 = nn.Conv2d(n_ch_out, n_ch_out, 3, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(n_ch_out, momentum=m)
        self.conv3 = nn.Conv2d(n_ch_out, n_ch_out, 1, padding=0, bias=True)
        self.bn3 = nn.BatchNorm2d(n_ch_out, momentum=m)
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        self.relu3 = nn.LeakyReLU()

    def forward(self, x):
        x = torch.cat((x[:,:,-1,:].unsqueeze(2),x,x[:,:,0,:].unsqueeze(2)),2)
        x = torch.cat((x[:,:,:,-1].unsqueeze(3),x,x[:,:,:,0].unsqueeze(3)),3)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = torch.cat((x[:,:,-1,:].unsqueeze(2),x,x[:,:,0,:].unsqueeze(2)),2)
        x = torch.cat((x[:,:,:,-1].unsqueeze(3),x,x[:,:,:,0].unsqueeze(3)),3)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x

#Up-sampling + batch normalization block
class Up_Bn2D(nn.Module):
    def __init__(self, n_ch):
        super(Up_Bn2D, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.bn = nn.BatchNorm2d(n_ch)

    def forward(self, x):
        x = self.bn(self.up(x))
        return x

class Pyramid2D(nn.Module):
    def __init__(self, ch_in=3, ch_step=8):
        super(Pyramid2D, self).__init__()

        self.cb1_1 = Conv_block2D(ch_in,ch_step)
        self.up1 = Up_Bn2D(ch_step)

        self.cb2_1 = Conv_block2D(ch_in,ch_step)
        self.cb2_2 = Conv_block2D(2*ch_step,2*ch_step)
        self.up2 = Up_Bn2D(2*ch_step)

        self.cb3_1 = Conv_block2D(ch_in,ch_step)
        self.cb3_2 = Conv_block2D(3*ch_step,3*ch_step)
        self.up3 = Up_Bn2D(3*ch_step)

        self.cb4_1 = Conv_block2D(ch_in,ch_step)
        self.cb4_2 = Conv_block2D(4*ch_step,4*ch_step)
        self.up4 = Up_Bn2D(4*ch_step)

        self.cb5_1 = Conv_block2D(ch_in,ch_step)
        self.cb5_2 = Conv_block2D(5*ch_step,5*ch_step)
        self.up5 = Up_Bn2D(5*ch_step)

        self.cb6_1 = Conv_block2D(ch_in,ch_step)
        self.cb6_2 = Conv_block2D(6*ch_step,6*ch_step)
        self.last_conv = nn.Conv2d(5*ch_step, 3, 1, padding=0, bias=True)#♣Modified

    def forward(self, z0,z1,z2,z3,z4):

        y = self.cb1_1(z4)
        y = self.up1(y)
        y = torch.cat((y,self.cb2_1(z3)),1)
        y = self.cb2_2(y)
        y = self.up2(y)
        y = torch.cat((y,self.cb3_1(z2)),1)
        y = self.cb3_2(y)
        y = self.up3(y)
        y = torch.cat((y,self.cb4_1(z1)),1)
        y = self.cb4_2(y)
        y = self.up4(y)
        y = torch.cat((y,self.cb5_1(z0)),1)
        y = self.cb5_2(y)
        #y = self.up5(y)
        #y = torch.cat((y,self.cb6_1(z[0])),1)
        #y = self.cb6_2(y)
        y = self.last_conv(y)
        return y

print('## Defining functions: done ##') 
      
#%%

#Gatys 

vgg19 = models.vgg19_bn(pretrained=True)

img = plt.imread('texture11.jpg')/255.
img = np.array(img, dtype=np.float32)
text = resize(img,(224,224,3))
plt.imshow(text)
text = torch.tensor(np.transpose(text,(2,0,1)),dtype=torch.float)

new_text = gatys(torch.unsqueeze(text,0),ite=2000,spectrum=True)

new_img = new_text.clone().cpu()[0]
plt.figure()
plt.imshow(np.transpose(new_img.detach().numpy(),(1,2,0)))
plt.imsave('Results_v2\gatys_spectrum_1e5.jpg',np.clip(np.transpose(new_img.detach().numpy(),(1,2,0)),0.,1.))


#%%

##################################################################################################################################
######################################################### EXECUTING CODE #########################################################
##################################################################################################################################

myNet = TextureNet()
#myNet = Pyramid2D()
myNet2 = copy.deepcopy(myNet)


vgg19 = models.vgg19_bn(pretrained=True)

noise0 = torch.rand((1,3,256,256),dtype=torch.float32)
noise1 = torch.rand((1,3,128,128),dtype=torch.float32)
noise2 = torch.rand((1,3,64,64),dtype=torch.float32)
noise3 = torch.rand((1,3,32,32),dtype=torch.float32)
noise4 = torch.rand((1,3,16,16),dtype=torch.float32)

out = myNet(noise0,noise1,noise2,noise3,noise4)
text = out.cpu().detach().numpy()
text = np.transpose(text[0],(1,2,0))
plt.imshow(text)

print('Loss =',texture_loss_bis(out.cuda()[0],preprocess_img(plt.imread('texture7.jpg')).cuda()[0]))
print('Valeur max de la texture générée =',np.max(text))
      
#%%

history = train(myNet, 16, 2000, 0.08, 'texture7.jpg')

myNetcpu = myNet.cpu()
out2 = myNetcpu(noise0,noise1,noise2,noise3,noise4)
text2 = out2.cpu().detach().numpy()
text2 = np.transpose(text2[0],(1,2,0))
plt.imshow(text2)
print(np.max(text2))

text_stretched = (text2 - np.min(text2))/(np.max(text2)-np.min(text2))
plt.figure()
plt.imshow(text_stretched)

#%%
#Try Ulyanov with VGG19 without batch normalisation
'''
vgg19 = models.vgg19(pretrained=True)
history = train(myNet2, 16, 250, 0.001, 'noise')

myNetcpu2 = myNet2.cpu()
out3 = myNetcpu2(noise0,noise1,noise2,noise3,noise4)
text4 = out3.cpu().detach().numpy()
text4 = np.transpose(text4[0],(1,2,0))
plt.imshow(text4)
print(np.max(text4))

text_stretched2 = (text4 - np.min(text4))/(np.max(text4)-np.min(text4))
plt.figure()
plt.imshow(text_stretched2)
'''
      
      
