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
import scipy.signal

print('Imports done')

#%% import the CNN

model = VGG19(include_top=True, pooling="avg", classes=1000)

model.summary()

layers_list=[1,4,7,12,17] #indexes of the layers we want to compute the gram matrix
layers_list_bis=["block1_conv1","block2_conv1","block3_conv1","block4_conv1","block5_conv1"]

layer=[[1],[1,4],[1,4,7],[1,4,7,12],[1,4,7,12,17]]
layer2=[[1],[4],[7],[12],[17]]
layer3=[layers_list]

#%% Testing the CNN
def Test_CNN(img_path):    
    img = image.load_img(img_path, target_size=(224, 224),interpolation="bicubic")
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    #show the image
    plt.imshow(img)
    plt.plot()    
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

#Gram matrix & Loss function definition
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

#Backend versions:
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
        num+=1
        loss += (1/(4*(depth*(width**2))**2))*K.sum((gram_Matrix_backend(activ1)[:,:]-gram_Matrix_backend(activ2[0][idx])[:,:])**2)#we assume that width=height
    return loss

def main_loss_backend(src_activ,layers):  
    return get_Tloss_backend(src_activ,layers)

#save image 'u' to the declared path
def save(path, u, mode=3):
    if (mode==1)or(mode==3):
        temp = (u-np.min(u))*254/(np.max(u)-np.min(u))
        plt.imsave(path+'.jpg',temp[0]/255)
    if (mode==2)or(mode==3):
        tempb=u
        tempb[u>255]=255
        tempb[u<0]=0
        plt.imsave(path+'b.jpg',tempb[0]/255)
    
#add 'blur_shape' blur to an image    
def blur_tensor(img_tensor,width,height,blur_shape):
    rouge = scipy.signal.convolve(img_tensor[0,:,:,0],1/9*np.ones(blur_shape,dtype=int),mode='same')
    vert = scipy.signal.convolve(img_tensor[0,:,:,1],1/9*np.ones(blur_shape,dtype=int),mode='same')
    bleu = scipy.signal.convolve(img_tensor[0,:,:,2],1/9*np.ones(blur_shape,dtype=int),mode='same')
    truc= np.zeros((width,height,3))
    truc[:,:,0]=rouge
    truc[:,:,1]=vert
    truc[:,:,2]=bleu
    return np.array([truc])

#load an image and convert it into a tensor
def load_tensor(img_path):
    new_img = image.load_img(img_path, target_size=(224, 224),interpolation="bicubic")
    new_tensor = image.img_to_array(new_img)                    # (height, width, channels)
    return np.expand_dims(new_tensor, axis=0)

#create white gaussian noise
def create_noise(moy,sigma,shape):
    x0 = np.random.normal(moy,sigma,shape)
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
    return x0_tensor

def create_grad_list(layer,active_func,src_tensor):
    src_act=active_func(src_tensor)
    output=[]
    for lyr in layer:
        output.append(K.function([model.input], K.gradients(main_loss_backend(src_act,lyr),model.input)[0]))
    return output

def grad_descent(u0,src_tensor,active_func,alpha_list,layer_list,iteration_list,plot_loss=False,plot_img=False,change_alpha=False,save_mode=0):
    u=u0
    src_act = active_func(src_tensor)
    loss_list=[]
    function_list=create_grad_list(layer_list,active_func,src_tensor)
    if ((len(alpha_list)!=len(layer_list))or(len(alpha_list)!=len(iteration_list))or(len(alpha_list)!=len(function_list))):
        raise Exception('Error: alpha_list, layer_list, iteration_list and function_list must have the same length.')
    if ((change_alpha) and (not(plot_loss))):
        raise Exception("Error: change_alpha can't be used without plot_loss (for the moment)")
    if plot_img:
        plt.figure()
        plt.imshow(u0[0]/255)
        plt.figure()
        plt.imshow(src_tensor[0]/255)
    for idx,layer in enumerate(layer_list):
        print('layer n°',idx+1,'/',len(layer_list))
        alpha=alpha_list[idx]
        func=function_list[idx]
        for i in range(iteration_list[idx]):
            if (i%10==0):
                print(".",end= " ")
            u = u - alpha*func(u)*255/np.max(func(u))
            if plot_loss:
                loss_list.append(np.log(texture_loss(layer,active_func(u),src_act)))
            if change_alpha and (i>2):
                if (loss_list[-1]>loss_list[-2]):
                    alpha=alpha/2
        if plot_img:
            plt.figure()
            plt.imshow(u[0]/255)
        save('layer'+str(idx),u,save_mode)
    if plot_loss:
        plt.figure()
        plt.plot(loss_list)
        plt.title('Loss for learning rate='+str(alpha_list)+';layers ='+str(layer_list)+';iterations='+str(iteration_list)+';change_alpha='+str(change_alpha))
        plt.show()
    return u[0]

#%% Define the source texture and white noise
src_img_tensor= load_tensor("texture3.jpg")
x0_tensor = create_noise(127,30,(224,224,3))
 
outputs = [layer.output for layer in model.layers]
active_func = K.function([model.input], [outputs])
src_act = active_func(src_img_tensor)

print("Start gradient descent")     

alphas=[1e-2,1e-2,1e-2,1e-2,1e-2]
iterations=[2000,400,200,200,200]
u0=blur_tensor(x0_tensor,224,224,(3,3))

result=grad_descent(u0,src_img_tensor,active_func,alphas,layer,iterations,True,True,True,save_mode=2)    


