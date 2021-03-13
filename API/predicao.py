#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 13:00:10 2021

@author: elilsonsantos
"""

from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
import pickle
import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

target_size = (224,224)

def carrega_modelo():

    model = tf.keras.models.load_model('model/my_model.hdf5')
  
    
    return model

_model = carrega_modelo()

def ler_imagem(image_encoded):
    
    pil_image = tf.keras.preprocessing.image.load_img(r'media/COVID-19 (1).png', target_size = (224,224))
    #pil_image = Image.open(BytesIO(image_encoded))
    
    return pil_image
    
def preprocessamento(image: Image.Image):
    
    #image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.asfarray(image)
    print(np.shape(image))
    image = np.expand_dims(image, axis = 0)
    image = tf.keras.applications.densenet.preprocess_input(image)
    
    return image
    

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

def predicao(image: Image.Image):
    
    teste = "teste"
    model = tf.keras.models.load_model('model/my_model.hdf5')
    
        
    #image = tf.keras.preprocessing.image.load_img(directory, target_size = (224,224))
    #print(type(image))
    image = tf.keras.preprocessing.image.img_to_array(image)
    print(np.shape(image))
    print(type(image))
    print(np.max(image))
    print(np.min(image))
    
    image = np.expand_dims(image, axis = 0)
    print(np.shape(image))
    
    predictions = model.predict(image)
    print(predictions)
    
    print(np.argmax(predictions))
    #prediction = np.argmax(predictions[0])
    

    return predictions[0]
