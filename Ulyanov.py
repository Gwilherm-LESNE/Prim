# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import time 
import torchvision.models as models
from skimage.transform import resize
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

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
    temp = torch.tensor(shape[1]*shape[2],dtype=torch.float)
    return result/temp

def texture_loss_bis(output,target):
    loss=torch.tensor(0.0,dtype=torch.float)
    index_list=[2,9,16,29,42]
    #index_list=[1,6,11,20,29]
    preprocess = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    for idx,_ in enumerate(index_list):
        act0 = activNet.get_activ(torch.unsqueeze(preprocess(output),dim=0),idx)
        act1 = activNet.get_activ(torch.unsqueeze(preprocess(target),dim=0),idx)
        temp = torch.pow(gram_matrix(act0)-gram_matrix(act1),2)
        loss = loss + torch.sum(temp)
    del index_list, preprocess, temp, act0, act1
    return loss

def custom_loss_bis(outputs,targets):#outputs & targets are 4D arrays, first dimension is batch size
    batch_loss = torch.tensor(0.0,dtype=torch.float)
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
    for idx,_ in enumerate(index_list):
        temp = torch.pow(gram_matrix(output[idx])-gram_matrix(activNet.get_activ(target,idx)),2)
        loss = loss + torch.sum(temp)
    if spectrum:
        loss += 1e-4 * spectrum_dist(target,output[-1])
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
            torch.nn.BatchNorm2d(8))
        
        self.block5_1.apply(weights_init)

        self.block6_1 = torch.nn.Sequential(
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
        
        self.block6_1.apply(weights_init)

        self.block5_2 = torch.nn.Sequential(
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
        
        self.block5_2.apply(weights_init)

        self.block4_2 = torch.nn.Sequential(
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
        
        self.block4_2.apply(weights_init)

        self.block3_2 = torch.nn.Sequential(
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
        
        self.block3_2.apply(weights_init)

        self.block2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(40,40,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(40),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(40,40,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(40),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(40,40,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(40),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(scale_factor=2,mode='nearest'),
            torch.nn.BatchNorm2d(40))
        
        self.block2_2.apply(weights_init)

        self.block1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(48,48,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(48),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(48,48,kernel_size=3,stride=1,padding=1, padding_mode= 'circular'),
            torch.nn.BatchNorm2d(48),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(48,48,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(48),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(48,3,kernel_size=1,stride=1,padding=0),
            torch.nn.BatchNorm2d(3),
            torch.nn.LeakyReLU())
        
        self.block1_2.apply(weights_init)

    def forward(self,x_0,x_1,x_2,x_3,x_4,x_5):
        x6 = self.block6_1(x_5)
        x5 = self.block5_1(x_4)
        x4 = self.block4_1(x_3)
        x3 = self.block3_1(x_2)
        x2 = self.block2_1(x_1)
        x1 = self.block1_1(x_0)

        x5 = torch.cat((x6,x5),dim=1)
        x5 = self.block5_2(x5)
        
        x4 = torch.cat((x5,x4),dim=1)
        x4 = self.block4_2(x4)
        
        x3 = torch.cat((x4,x3),dim=1)
        x3 = self.block3_2(x3)
        
        x2 = torch.cat((x3,x2),dim=1)
        x2 = self.block2_2(x2)
        
        x1 = torch.cat((x2,x1),dim=1)
        out = self.block1_2(x1)

        del x1,x2,x3,x4,x5,x6
        return out
    
    def cuda(self):
      self.block6_1.cuda()
      self.block5_1.cuda()
      self.block4_1.cuda()
      self.block3_1.cuda()
      self.block2_1.cuda()
      self.block1_1.cuda()
      self.block5_2.cuda()
      self.block4_2.cuda()
      self.block3_2.cuda()
      self.block2_2.cuda()
      self.block1_2.cuda()
    
class NoiseTextureDataset(Dataset):
    
    def __init__(self, size, texture_path):
        self.size = size
        
        prep = transforms.Compose([transforms.ToTensor(),])
        
        if texture_path == 'noise':
          text = torch.rand((3,256,256),dtype=torch.float)
        else:
          img = Image.open(texture_path)
          text = prep(img)

        #preprocess = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        #text = preprocess(text)
        self.texture = torch.unsqueeze(text,0)
            
        self.noise0 = torch.rand((size,1,3,256,256),dtype=torch.float)
        self.noise1 = torch.rand((size,1,3,128,128),dtype=torch.float)
        self.noise2 = torch.rand((size,1,3,64,64),dtype=torch.float)
        self.noise3 = torch.rand((size,1,3,32,32),dtype=torch.float)
        self.noise4 = torch.rand((size,1,3,16,16),dtype=torch.float)
        self.noise5 = torch.rand((size,1,3,8,8),dtype=torch.float)
        
        self.sample = []
        for idx in range(size):
            self.sample.append({'noise0': self.noise0[idx].cuda(),
                  'noise1': self.noise1[idx].cuda(),
                  'noise2': self.noise2[idx].cuda(),
                  'noise3': self.noise3[idx].cuda(),
                  'noise4': self.noise4[idx].cuda(),
                  'noise5': self.noise5[idx].cuda(),
                  'texture': self.texture.cuda()})
    

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()           
        return self.sample[idx]

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

def epoch(data, model, optimizer, train_history, n_iter):
    start_time = time.time()
    total_train_loss = torch.tensor(0.,dtype=torch.float)
    count = torch.tensor(0.,dtype=torch.float)
    for i, sample in enumerate(data):
        noise0 = sample['noise0']
        noise1 = sample['noise1']
        noise2 = sample['noise2']
        noise3 = sample['noise3']
        noise4 = sample['noise4']
        noise5 = sample['noise5']
        targets = sample['texture']
        
        # forward + backward
        optimizer.zero_grad()
        outputs = model(noise0,noise1,noise2,noise3,noise4,noise5)
        #Tried a clamp here
        loss = custom_loss_bis(outputs, targets)
        loss.backward()        
        for p in model.parameters(recurse=True):
            if (p.grad != None):
              p.grad /= torch.norm(p.grad, p=1, dim=0)
        optimizer.step()
        total_train_loss += loss.detach().item()
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
        count += torch.tensor(1.0,dtype=torch.float)
        del noise0, noise1, noise2, noise3, noise4, noise5, targets, loss, p, outputs
    
    if (n_iter%10 == 9)or True:
      print("\n==== EPOCH "+str(n_iter+1)+" ====\n")
      print("train_loss: {:.5f} took: {:.2f}s".format(total_train_loss / count,time.time() - start_time))
    train_history.append(total_train_loss / count)
    del total_train_loss, count, start_time

def train(net, batch_size, n_epochs, learning_rate, texture_path):
    
    print("====== HYPERPARAMETERS ======")
    print("batch_size=", batch_size)
    print("n_epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 29)
    
    model = net
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(recurse=True),lr=learning_rate)
    #optimizer = torch.optim.LBFGS(model.parameters())
    lr_sched = torch.optim.lr_scheduler.ExponentialLR (optimizer, gamma=0.95)

    for p in model.parameters():
      p.requires_grad=True
    
    dataset = NoiseTextureDataset(batch_size, texture_path)
    train_history = []

    # On itère sur les epochs
    for i in range(n_epochs):          
        epoch(dataset, model, optimizer, train_history, i)
        #To use if you want to update the learning rate
        if (i%100 == 1) and (i>100):
            lr_sched.step()
    return train_history

def gatys(text,ite=20,spectrum=False):
    x0 = torch.rand((1,3,224,224),dtype=torch.float).cuda()
    x0.requires_grad = True
    text=text.cuda()

    optimizer = torch.optim.LBFGS([x0])
    #optimizer = torch.optim.Adam([x0],lr=1)
    activNet = ActivNet([2,9,16,29,42])
    activNet.cuda()
    activNet.eval()

#    new_img = x0.clone().cpu()[0]
#    plt.figure()
#    plt.imshow(np.transpose(new_img.detach().numpy(),(1,2,0)))

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

postpb = transforms.Compose([transforms.ToPILImage()])
def postp(t): # to clip results in the range [0,1]
    t[t>1] = 1
    t[t<0] = 0
    img = postpb(t)
    return img

print('## Defining functions: done ##') 
      
#%%

#Gatys 

#vgg19 = models.vgg19_bn(pretrained=True)
#activNet = ActivNet(index_list=[2,9,16,29,42])
#activNet.cuda()
#activNet.eval()
#img = plt.imread('texture11.jpg')/255.
#img = np.array(img, dtype=np.float32)
#text = resize(img,(224,224,3))
#plt.imshow(text)
#text = torch.tensor(np.transpose(text,(2,0,1)),dtype=torch.float)
#
#new_text = gatys(torch.unsqueeze(text,0),ite=2000,spectrum=True)
#
#new_img = new_text.clone().cpu()[0]
#plt.figure()
#plt.imshow(np.transpose(new_img.detach().numpy(),(1,2,0)))
#plt.imsave('Results_v2\gatys_spectrum_1e5.jpg',np.clip(np.transpose(new_img.detach().numpy(),(1,2,0)),0.,1.))


#%%
##################################################################################################################################
######################################################### EXECUTING CODE #########################################################
##################################################################################################################################

myNet = TextureNet()
#myNet = torch.load('TextureNet.pt')

#vgg19=torch.load('./vgg19.pt')
vgg19 = models.vgg19_bn(pretrained=True)

activNet = ActivNet(index_list=[2,9,16,29,42]) # if normal, index_list= [1,6,11,20,29]. if batch norm : index_list = [2,9,16,29,42]
activNet.cuda()
activNet.eval()
      
#%%
#Train

history = train(myNet, 16, 2000, 0.1, 'texture7.jpg')

torch.save(myNet,'TextureNet7_bn.pt')

#%%
#Generate samples

for i in range(5):
    z0 = torch.rand((1,3,256,256),dtype=torch.float32).cuda()
    z1 = torch.rand((1,3,128,128),dtype=torch.float32).cuda()
    z2 = torch.rand((1,3,64,64),dtype=torch.float32).cuda()
    z3 = torch.rand((1,3,32,32),dtype=torch.float32).cuda()
    z4 = torch.rand((1,3,16,16),dtype=torch.float32).cuda()
    z5 = torch.rand((1,3,8,8),dtype=torch.float32).cuda()
    out = myNet(z0,z1,z2,z3,z4,z5)
    text = out.cpu().detach()
    img = postp(text[0])
    fname = './Results_v2/sample'+str(i)+'.jpg'
    img.save(fname)      
      
