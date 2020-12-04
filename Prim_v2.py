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
import scipy.fft
import tensorflow_probability as tfp

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

def get_activ(layer_name,archi='vgg19'):
    if archi=='vgg19':
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in layer_name]
        model = tf.keras.Model([vgg.input], outputs)
    elif archi=='inceptionV3':
        incep=tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        incep.trainable = False
        outputs = [incep.get_layer(name).output for name in layer_name]
        model = tf.keras.Model([incep.input], outputs)
    return model

def gram_matrix(x):
  result = tf.linalg.einsum('bijc,bijd->bcd', x, x)
  shape = tf.shape(x)
  temp = tf.cast(shape[1]*shape[2], tf.float32)
  return result/temp

def texture_loss(x,texture,model_act,archi='vgg19'):
    if archi=='vgg19':
        noise_activ = model_act(tf.keras.applications.vgg19.preprocess_input(x*255))
        text_activ = model_act(tf.keras.applications.vgg19.preprocess_input(texture*255))
    elif archi=='inceptionV3':
        noise_activ = model_act(tf.keras.applications.inception_v3.preprocess_input(x*255))
        text_activ = model_act(tf.keras.applications.inception_v3.preprocess_input(texture*255))
    loss= tf.constant(0.0)
    for idx,_ in enumerate(noise_activ):
        temp = tf.square(gram_matrix(noise_activ[idx])-gram_matrix(text_activ[idx]))
        loss += tf.math.reduce_sum(temp)
    return loss

def grad_descent(x0,text,layer_name = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1'],iterations=2000,plot_loss=True,lr=0.2,save_img=False,archi='vgg19'):
    x=tf.Variable(x0)
    loss_list=[]
    model_act = get_activ(layer_name,archi)
    opt = tf.optimizers.Adam(learning_rate=lr, beta_1=0.99, epsilon=1e-1)
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss=texture_loss(x,text,model_act,archi)
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

def spectrum_dist(src_img,ipt_img): #inputs are 4d tensors of shape (1,w,h,c)
    src= tf.cast(src_img[0],dtype=tf.complex128)
    ipt= tf.cast(ipt_img[0],dtype=tf.complex128)
    fi = tf.signal.fft3d(src)
    fi_hat = tf.signal.fft3d(ipt)
    prod = tf.math.multiply(fi_hat,tf.math.conj(fi))
    norm = tf.math.divide(prod,tf.cast(tf.math.abs(prod),dtype=tf.complex64)) 
    i_tilde = tf.signal.ifft3d(tf.math.multiply(norm,fi))
    output = tf.expand_dims(i_tilde, axis=0)
    return output

def spectrum_dist_bis(src_img,ipt_img): #inputs are 4d tensors of shape (1,w,h,c)
    src= tf.cast(src_img[0],dtype=tf.complex128).numpy()
    ipt= tf.cast(ipt_img[0],dtype=tf.complex128).numpy()
    fi = scipy.fft.fftn(src)
    fi_hat = scipy.fft.fftn(ipt)
    prod = np.multiply(fi_hat,np.conj(fi))
    norm = np.divide(prod,np.absolute(prod)) 
    i_tilde = scipy.fft.ifftn(np.multiply(norm,fi))
    output = np.expand_dims(i_tilde, axis=0)
    return np.linalg.norm(output)

def spectrum_loss(x,texture,model_act,archi='vgg19',beta=1e4):
    if archi=='vgg19':
        noise_activ = model_act(tf.keras.applications.vgg19.preprocess_input(x*255))
        text_activ = model_act(tf.keras.applications.vgg19.preprocess_input(texture*255))
    elif archi=='inceptionV3':
        noise_activ = model_act(tf.keras.applications.inception_v3.preprocess_input(x*255))
        text_activ = model_act(tf.keras.applications.inception_v3.preprocess_input(texture*255))
    loss= tf.constant(0.0)
    for idx,_ in enumerate(noise_activ):
        temp = tf.square(gram_matrix(noise_activ[idx])-gram_matrix(text_activ[idx]))
        loss += tf.math.reduce_sum(temp)*1e-10
    loss = tf.constant(0.0)
    loss += beta*tf.cast(tf.Variable(spectrum_dist_bis(texture,x)),dtype=tf.float64)
    return loss


def spectrum_grad_descent(x0,text,beta=1e-3,layer_name = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1'],iterations=2000,plot_loss=True,lr=0.2,save_img=False,archi='vgg19'):
    x=tf.Variable(x0)
    loss_list=[]
    model_act = get_activ(layer_name,archi)
    opt = tf.optimizers.Adam(learning_rate=lr, beta_1=0.99, epsilon=1e-1)
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss=spectrum_loss(x,text,model_act,archi,beta)
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
        temp_str='./Results_v2/spectrum_ite=%sk_lr=%s.jpg'%(iterations//1000,lr)
        plt.imsave(temp_str,tensor_to_array(x))
    return x

#L-BFGS Functions
def lbfgs_loss(ipt_tensor):
  ipt = tf.reshape(ipt_tensor, (1,224,224,3))
  text13=preprocess_img('texture13.jpg')
  model_act = get_activ(layer_name=['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1'])
  return texture_loss(ipt,text13,model_act)

def grad_and_val(ipt_tensor):
  return tfp.math.value_and_gradient(lbfgs_loss,ipt_tensor,use_gradient_tape=False)

#%%
x0 = tf.constant(np.random.randn(1,224,224,3)/9 + 0.5) # (1,224,224,3)/9+0.5 to be in [0,1] range 

layer_name = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
layer_name2 = ['block1_pool','block2_pool','block3_pool','block4_pool','block5_pool']
layer_name3 = ['max_pooling2d','max_pooling2d_1','mixed1','mixed3','mixed5','mixed7','mixed9']

incep_layer_names=['input_1',
 'conv2d',
 'batch_normalization',
 'activation',
 'conv2d_1',
 'batch_normalization_1',
 'activation_1',
 'conv2d_2',
 'batch_normalization_2',
 'activation_2',
 'max_pooling2d',
 'conv2d_3',
 'batch_normalization_3',
 'activation_3',
 'conv2d_4',
 'batch_normalization_4',
 'activation_4',
 'max_pooling2d_1',
 'conv2d_8',
 'batch_normalization_8',
 'activation_8',
 'conv2d_6',
 'conv2d_9',
 'batch_normalization_6',
 'batch_normalization_9',
 'activation_6',
 'activation_9',
 'average_pooling2d',
 'conv2d_5',
 'conv2d_7',
 'conv2d_10',
 'conv2d_11',
 'batch_normalization_5',
 'batch_normalization_7',
 'batch_normalization_10',
 'batch_normalization_11',
 'activation_5',
 'activation_7',
 'activation_10',
 'activation_11',
 'mixed0',
 'conv2d_15',
 'batch_normalization_15',
 'activation_15',
 'conv2d_13',
 'conv2d_16',
 'batch_normalization_13',
 'batch_normalization_16',
 'activation_13',
 'activation_16',
 'average_pooling2d_1',
 'conv2d_12',
 'conv2d_14',
 'conv2d_17',
 'conv2d_18',
 'batch_normalization_12',
 'batch_normalization_14',
 'batch_normalization_17',
 'batch_normalization_18',
 'activation_12',
 'activation_14',
 'activation_17',
 'activation_18',
 'mixed1',
 'conv2d_22',
 'batch_normalization_22',
 'activation_22',
 'conv2d_20',
 'conv2d_23',
 'batch_normalization_20',
 'batch_normalization_23',
 'activation_20',
 'activation_23',
 'average_pooling2d_2',
 'conv2d_19',
 'conv2d_21',
 'conv2d_24',
 'conv2d_25',
 'batch_normalization_19',
 'batch_normalization_21',
 'batch_normalization_24',
 'batch_normalization_25',
 'activation_19',
 'activation_21',
 'activation_24',
 'activation_25',
 'mixed2',
 'conv2d_27',
 'batch_normalization_27',
 'activation_27',
 'conv2d_28',
 'batch_normalization_28',
 'activation_28',
 'conv2d_26',
 'conv2d_29',
 'batch_normalization_26',
 'batch_normalization_29',
 'activation_26',
 'activation_29',
 'max_pooling2d_2',
 'mixed3',
 'conv2d_34',
 'batch_normalization_34',
 'activation_34',
 'conv2d_35',
 'batch_normalization_35',
 'activation_35',
 'conv2d_31',
 'conv2d_36',
 'batch_normalization_31',
 'batch_normalization_36',
 'activation_31',
 'activation_36',
 'conv2d_32',
 'conv2d_37',
 'batch_normalization_32',
 'batch_normalization_37',
 'activation_32',
 'activation_37',
 'average_pooling2d_3',
 'conv2d_30',
 'conv2d_33',
 'conv2d_38',
 'conv2d_39',
 'batch_normalization_30',
 'batch_normalization_33',
 'batch_normalization_38',
 'batch_normalization_39',
 'activation_30',
 'activation_33',
 'activation_38',
 'activation_39',
 'mixed4',
 'conv2d_44',
 'batch_normalization_44',
 'activation_44',
 'conv2d_45',
 'batch_normalization_45',
 'activation_45',
 'conv2d_41',
 'conv2d_46',
 'batch_normalization_41',
 'batch_normalization_46',
 'activation_41',
 'activation_46',
 'conv2d_42',
 'conv2d_47',
 'batch_normalization_42',
 'batch_normalization_47',
 'activation_42',
 'activation_47',
 'average_pooling2d_4',
 'conv2d_40',
 'conv2d_43',
 'conv2d_48',
 'conv2d_49',
 'batch_normalization_40',
 'batch_normalization_43',
 'batch_normalization_48',
 'batch_normalization_49',
 'activation_40',
 'activation_43',
 'activation_48',
 'activation_49',
 'mixed5',
 'conv2d_54',
 'batch_normalization_54',
 'activation_54',
 'conv2d_55',
 'batch_normalization_55',
 'activation_55',
 'conv2d_51',
 'conv2d_56',
 'batch_normalization_51',
 'batch_normalization_56',
 'activation_51',
 'activation_56',
 'conv2d_52',
 'conv2d_57',
 'batch_normalization_52',
 'batch_normalization_57',
 'activation_52',
 'activation_57',
 'average_pooling2d_5',
 'conv2d_50',
 'conv2d_53',
 'conv2d_58',
 'conv2d_59',
 'batch_normalization_50',
 'batch_normalization_53',
 'batch_normalization_58',
 'batch_normalization_59',
 'activation_50',
 'activation_53',
 'activation_58',
 'activation_59',
 'mixed6',
 'conv2d_64',
 'batch_normalization_64',
 'activation_64',
 'conv2d_65',
 'batch_normalization_65',
 'activation_65',
 'conv2d_61',
 'conv2d_66',
 'batch_normalization_61',
 'batch_normalization_66',
 'activation_61',
 'activation_66',
 'conv2d_62',
 'conv2d_67',
 'batch_normalization_62',
 'batch_normalization_67',
 'activation_62',
 'activation_67',
 'average_pooling2d_6',
 'conv2d_60',
 'conv2d_63',
 'conv2d_68',
 'conv2d_69',
 'batch_normalization_60',
 'batch_normalization_63',
 'batch_normalization_68',
 'batch_normalization_69',
 'activation_60',
 'activation_63',
 'activation_68',
 'activation_69',
 'mixed7',
 'conv2d_72',
 'batch_normalization_72',
 'activation_72',
 'conv2d_73',
 'batch_normalization_73',
 'activation_73',
 'conv2d_70',
 'conv2d_74',
 'batch_normalization_70',
 'batch_normalization_74',
 'activation_70',
 'activation_74',
 'conv2d_71',
 'conv2d_75',
 'batch_normalization_71',
 'batch_normalization_75',
 'activation_71',
 'activation_75',
 'max_pooling2d_3',
 'mixed8',
 'conv2d_80',
 'batch_normalization_80',
 'activation_80',
 'conv2d_77',
 'conv2d_81',
 'batch_normalization_77',
 'batch_normalization_81',
 'activation_77',
 'activation_81',
 'conv2d_78',
 'conv2d_79',
 'conv2d_82',
 'conv2d_83',
 'average_pooling2d_7',
 'conv2d_76',
 'batch_normalization_78',
 'batch_normalization_79',
 'batch_normalization_82',
 'batch_normalization_83',
 'conv2d_84',
 'batch_normalization_76',
 'activation_78',
 'activation_79',
 'activation_82',
 'activation_83',
 'batch_normalization_84',
 'activation_76',
 'mixed9_0',
 'concatenate',
 'activation_84',
 'mixed9',
 'conv2d_89',
 'batch_normalization_89',
 'activation_89',
 'conv2d_86',
 'conv2d_90',
 'batch_normalization_86',
 'batch_normalization_90',
 'activation_86',
 'activation_90',
 'conv2d_87',
 'conv2d_88',
 'conv2d_91',
 'conv2d_92',
 'average_pooling2d_8',
 'conv2d_85',
 'batch_normalization_87',
 'batch_normalization_88',
 'batch_normalization_91',
 'batch_normalization_92',
 'conv2d_93',
 'batch_normalization_85',
 'activation_87',
 'activation_88',
 'activation_91',
 'activation_92',
 'batch_normalization_93',
 'activation_85',
 'mixed9_1',
 'concatenate_1',
 'activation_93',
 'mixed10']

text11=preprocess_img('texture11.jpg')
#%%
print("Start gradient descent")
with tf.device("/device:GPU:0"):
    
    var0 = tf.Variable(x0)
    var1 = tf.Variable(x0)
    plt.figure()
    plt.imshow(tensor_to_image(var0))
    
    start_time = time.time()
    out=grad_descent(var0, text11, iterations=40, plot_loss=True, lr=0.02, save_img=False)
    out_img = tensor_to_image(out)
    print(" --- %s seconds ---" % (time.time() - start_time))
    plt.figure()
    plt.imshow(out_img)
    
    plt.figure()
    plt.imshow(tensor_to_image(var1))

    start_time = time.time()
    new_out=spectrum_grad_descent(var1, text11, beta=1e8, iterations=40, plot_loss=True, lr=0.02, save_img=False)
    new_out_img = tensor_to_image(new_out)
    print(" --- %s seconds ---" % (time.time() - start_time))
    plt.figure()
    plt.imshow(new_out_img)
    
    
    #L-BFGS PART:
    result = tfp.optimizer.lbfgs_minimize(grad_and_val,var0,max_iterations=10)
    print(result.num_iterations)
    out_img = tensor_to_image(tf.reshape(result.position,(1,224,224,3)))
    plt.figure()
    plt.imshow(out_img)
