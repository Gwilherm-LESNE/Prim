# -*- coding: utf-8 -*-
#imports

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import scipy.optimize

print('Imports done')

#%% import the CNN

model = VGG19(include_top=True, pooling="avg", classes=1000)

model.summary()

#%% Testing the CNN

img_path = "tiger.jpg"

img = image.load_img(img_path, target_size=(224, 224),interpolation="bicubic")
img_tensor = image.img_to_array(img)                    # (height, width, channels)
img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)

#show the image

#plt.imshow(img)
#plt.plot()

#predict the class of the image
prediction = model.predict(img_tensor)
pred = np.argmax(prediction, axis = 1)
print("class number:", pred[0])

#convert index of the prediction into a readable label
with open("imagenet_labels.txt") as file_in:
    labels = []
    for line in file_in:
        labels.append(line)
print(labels[pred[0]+1])

#%%Gram matrix & Loss function definition

def gram_Matrix(activation):
    act=activation[0]
    s = act.shape
    temp=np.reshape(act,(s[0]*s[1],s[2]))
    new_gram= np.dot(np.transpose(temp),temp)
    return new_gram

def get_Tloss(act1,act2,layers_list):
    loss=0
    for idx in layers_list:
        width = np.shape(act1[0][idx])[1]
        depth = np.shape(act1[0][idx])[3]
        loss+= (1/(4*(depth*(width**2))**2))*np.sum((gram_Matrix(act1[0][idx])[:,:]-gram_Matrix(act2[0][idx])[:,:])**2)#we assume that width=height
    return loss

def texture_loss(layers_list,act1,act2):
    return get_Tloss(act1,act2,layers_list)

#%% Define the source texture

layers_list=[1,4,7,12,17] #indexes of the layers we want to compute the gram matrix
layers_list_bis=["block1_conv1","block2_conv1","block3_conv1","block4_conv1","block5_conv1"]

src_img_path = "texture1.jpg"  #There is 5 textures for the moment.

src_img = image.load_img(src_img_path, target_size=(224, 224),interpolation="bicubic")
src_img_tensor = image.img_to_array(src_img)                    # (height, width, channels)
src_img_tensor = np.expand_dims(src_img_tensor, axis=0)

outputs = [layer.output for layer in model.layers]
active_func = K.function([model.input], [outputs])
src_act = active_func(src_img_tensor)

#create white gaussian noise
x0 = np.random.normal(127,30,(224,224,3))
x0= np.floor(x0)
for i in range(np.shape(x0)[0]):
    for j in range(np.shape(x0)[1]):
        for k in range(np.shape(x0)[2]):
            if x0[i,j,k]>255.:
                x0[i,j,k]=255.
            elif x0[i,j,k]<0.:
                x0[i,j,k]=0.
x0_tensor = image.img_to_array(x0)
x0_tensor = np.expand_dims(x0_tensor, axis=0)
ipt_act = active_func(x0_tensor)

#%%

def gram_Matrix_backend(activation):
    act=activation[0]
    s = act.shape
    temp=K.reshape(act,(s[0]*s[1],s[2]))
    new_gram= K.dot(K.transpose(temp),temp)
    return new_gram
    
def get_Tloss_backend(activ2,layers_list):
    loss=K.constant(0)
    num=1
    for idx in layers_list:
        activ1= model.layers[idx].output
        width = np.shape(activ2[0][idx])[1]
        depth = np.shape(activ2[0][idx])[3]
        print("step",num,"/",len(layers_list))
        num+=1
        loss += (1/(4*(depth*(width**2))**2))*K.sum((gram_Matrix_backend(activ1)[:,:]-gram_Matrix_backend(activ2[0][idx])[:,:])**2)#we assume that width=height
    return loss

def main_loss_backend(src_activ,layers):  
    return get_Tloss_backend(src_activ,layers)

#def fun(alpha):#alpha is a 1D array of shape = (1,)
#    act1 = active_func(x0_tensor-alpha[0]*iterate(x0_tensor))
#    return texture_loss(layers_list,act1,src_act)

#%%
print("Start gradient computation")

deriv=K.gradients(main_loss_backend(src_act,layers_list),model.input)[0]

iterate = K.function([model.input], deriv)

print(np.shape(iterate(x0_tensor)))
print(np.shape(x0_tensor))

#result=scipy.optimize.minimize(fun,np.array([0]),method='L-BFGS-B')#too long
#%%
print("Start gradient descent")

u0=x0_tensor
u=u0
alpha=1e-2
N=70
Loss_list=[]
for i in range(N):
    print('step',i+1,'/',N)
    u = u - alpha*iterate(u)*255/np.max(iterate(u))
    u = (u-np.min(u))*255/(np.max(u)-np.min(u)) #streching d'histogramme
    Loss_list.append(np.log(texture_loss(layers_list,active_func(u),src_act)))
    
print('max=',np.max(u[0]))
print('min=',np.min(u[0]))
u0 = (u0-np.min(u0))*255/(np.max(u0)-np.min(u0))
u = (u-np.min(u))*255/(np.max(u)-np.min(u))    
plt.imshow((u0)[0]/255) 
plt.figure()
plt.imshow(u[0]/255)
plt.figure()
plt.imshow(src_img_tensor[0]/255)
plt.figure()
plt.plot(Loss_list)









