# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:59:53 2020

@author: ROG
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time 
import PIL.Image

print('Imports Done')

#%%
def preprocess_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)    
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, axis=0)
    x = tf.image.resize(img, (224, 224))
    return x
    
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def tensor_to_array(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return tensor

def get_activ(layer_name):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_name]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(x):
  result = tf.linalg.einsum('bijc,bijd->bcd', x, x)
  shape = tf.shape(x)
  temp = tf.cast(shape[1]*shape[2], tf.float32)
  return result/temp

def texture_loss(x,texture,model_act):
    noise_activ = model_act(tf.keras.applications.vgg19.preprocess_input(x*255))
    text_activ = model_act(tf.keras.applications.vgg19.preprocess_input(texture*255))
    loss= tf.constant(0.0)
    for idx,_ in enumerate(noise_activ):
        temp = tf.square(gram_matrix(noise_activ[idx])-gram_matrix(text_activ[idx]))
        loss += tf.math.reduce_sum(temp)
    return loss

def grad_descent(x0,text,layer_name = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1'],iterations=2000,plot_loss=True,lr=0.2,save_img=False):
    x=tf.Variable(x0)
    loss_list=[]
    model_act = get_activ(layer_name)
    opt = tf.optimizers.Adam(learning_rate=lr, beta_1=0.99, epsilon=1e-1)
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss=texture_loss(x,text,model_act)
        loss_list.append(np.log(loss))
        grad= tape.gradient(loss,x)
        opt.apply_gradients([(grad, x)])
        x.assign(tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
        print(".",end="")
    print(".")
    if (plot_loss):
        plt.figure()
        plt.plot(loss_list)
    if (save_img):
        temp_str='./Results_v2/ite=%sk_lr=%s.jpg'%(iterations//1000,lr)
        plt.imsave(temp_str,tensor_to_array(x))
    return x

#%%
    

def grad_descent1(x0,text,layer_name = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1'],iterations=2000,plot_loss=True,lr=0.2,save_img=False):
    x=tf.Variable(x0)
    loss_list=[]
    model_act = get_activ(layer_name)
    opt = tf.optimizers.Adam(learning_rate=lr, beta_1=0.99, epsilon=1e-1)
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss=texture_loss(x,text,model_act)
        loss_list.append(np.log(loss))
        grad= tape.gradient(loss,x)
        opt.apply_gradients([(grad, x)])
        x.assign(tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
        print(".",end="")
    print(".")
    if (plot_loss):
        plt.figure()
        plt.plot(loss_list)
    if (save_img):
        temp_str='./Results_v2/1ite=%sk_lr=%s.jpg'%(iterations//100,lr)
        plt.imsave(temp_str,tensor_to_array(x))
    return x

def grad_descent2(x0,text,layer_name = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1'],iterations=2000,plot_loss=True,lr=0.2,save_img=False):
    x=tf.Variable(x0)
    loss_list=[]
    model_act = get_activ(layer_name)
    opt = tf.optimizers.Adam(learning_rate=lr, beta_1=0.99, epsilon=1e-1)
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss=texture_loss(x,text,model_act)
        loss_list.append(np.log(loss))
        grad= tape.gradient(loss,x)
        opt.apply_gradients([(grad, x)])
        x.assign(tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
        print(".",end="")
    print(".")
    if (plot_loss):
        plt.figure()
        plt.plot(loss_list)
    if (save_img):
        temp_str='./Results_v2/2ite=%sk_lr=%s.jpg'%(iterations//100,lr)
        plt.imsave(temp_str,tensor_to_array(x))
    return x

def grad_descent3(x0,text,layer_name = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1'],iterations=2000,plot_loss=True,lr=0.2,save_img=False):
    x=tf.Variable(x0)
    loss_list=[]
    model_act = get_activ(layer_name)
    opt = tf.optimizers.Adam(learning_rate=lr, beta_1=0.99, epsilon=1e-1)
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss=texture_loss(x,text,model_act)
        loss_list.append(np.log(loss))
        grad= tape.gradient(loss,x)
        opt.apply_gradients([(grad, x)])
        x.assign(tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
        print(".",end="")
    print(".")
    if (plot_loss):
        plt.figure()
        plt.plot(loss_list)
    if (save_img):
        temp_str='./Results_v2/3ite=%sk_lr=%s.jpg'%(iterations//100,lr)
        plt.imsave(temp_str,tensor_to_array(x))
    return x

def grad_descent4(x0,text,layer_name = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1'],iterations=2000,plot_loss=True,lr=0.2,save_img=False):
    x=tf.Variable(x0)
    loss_list=[]
    model_act = get_activ(layer_name)
    opt = tf.optimizers.Adam(learning_rate=lr, beta_1=0.99, epsilon=1e-1)
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss=texture_loss(x,text,model_act)
        loss_list.append(np.log(loss))
        grad= tape.gradient(loss,x)
        opt.apply_gradients([(grad, x)])
        x.assign(tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
        print(".",end="")
    print(".")
    if (plot_loss):
        plt.figure()
        plt.plot(loss_list)
    if (save_img):
        temp_str='./Results_v2/4ite=%sk_lr=%s.jpg'%(iterations//100,lr)
        plt.imsave(temp_str,tensor_to_array(x))
    return x
#%%
x0=tf.constant(np.random.randn(1,224,224,3)/4.5)
layer_name = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
layer_name2 = ['block1_pool','block2_pool','block3_pool','block4_pool','block5_pool']
text=preprocess_img('texture7.jpg')
text2=preprocess_img('texture6.jpg')
text3=preprocess_img('texture3.jpg')
text4=preprocess_img('texture4.jpg')

print("Start gradient descent")
with tf.device("/device:GPU:0"):
    
#    start_time = time.time()
#    out = grad_descent(x0,text)
#    out_img = tensor_to_image(out)
#    print(" --- %s seconds ---" % (time.time() - start_time))
#    plt.figure()
#    plt.imshow(out_img)
    
    
    x0=tf.constant(np.random.randn(1,224,224,3)/4.5)
    grad_descent2(x0,text2,iterations=4000,plot_loss=False,lr=0.2,save_img=True)
    
    x0=tf.constant(np.random.randn(1,224,224,3)/4.5)
    grad_descent2(x0,text2,iterations=4000,plot_loss=False,lr=0.1,save_img=True)
    
    x0=tf.constant(np.random.randn(1,224,224,3)/4.5)
    grad_descent2(x0,text2,iterations=4000,plot_loss=False,lr=0.05,save_img=True)
    
    x0=tf.constant(np.random.randn(1,224,224,3)/4.5)
    grad_descent2(x0,text2,iterations=4000,plot_loss=False,lr=0.02,save_img=True)
    
    print('save')
    
    x0=tf.constant(np.random.randn(1,224,224,3)/4.5)
    grad_descent2(x0,text2,iterations=6000,plot_loss=False,lr=0.2,save_img=True)
    
    x0=tf.constant(np.random.randn(1,224,224,3)/4.5)
    grad_descent2(x0,text2,iterations=6000,plot_loss=False,lr=0.1,save_img=True)
    
    x0=tf.constant(np.random.randn(1,224,224,3)/4.5)
    grad_descent2(x0,text2,iterations=6000,plot_loss=False,lr=0.05,save_img=True)
    
    x0=tf.constant(np.random.randn(1,224,224,3)/4.5)
    grad_descent2(x0,text2,iterations=6000,plot_loss=False,lr=0.02,save_img=True)

    
    
    

