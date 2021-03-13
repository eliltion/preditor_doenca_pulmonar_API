#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 01:24:31 2021

@author: elilsonsantos
"""

from io import BytesIO

import numpy as np
import tensorflow as tf
import keras
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import decode_predictions



input_shape = (224,224)

def load_model():
    #model = tf.keras.applications.MobileNetV2(weights="imagenet")
    model = tf.keras.models.load_model('model/my_model.hdf5')
    print("Model loaded")
    return model


def predict(image: Image.Image, model):
   
    if model is None:
        model = load_model()
        
    print(type(image))
    print(np.shape(image))

    #image = image.resize(input_shape)
    #image = np.asarray(image)
    #image = tf.keras.preprocessing.image.lo
    image = np.asarray(image.resize((224, 224)))
    image = tf.keras.preprocessing.image.img_to_array(image)
    print(np.shape(image))
    print(type(image))
    print(np.max(image))
    print(np.min(image))
    
    image = np.expand_dims(image, axis = 0)
    print(np.shape(image))  
    
    #image = tf.keras.applications.densenet.preprocess_input(image)
    
    image = image / 127.5 - 1.0
    
    
    print(np.max(image))
    print(np.min(image))
   
    #image = image / 127.5 - 1.0

    #result = decode_predictions(model.predict(image), 2)[0]
    
    result = model.predict(image)
    
    
    #print(result[0])
    
    #response = []
    #for i, res in enumerate(result):
    #    resp = {}
    #    resp[confidence] = f"{res[i]*100:0.2f} %"
        #resp["confidence"] = f"{res[2]*100:0.2f} %"

   #     response.append(resp)
        
   # print(response)

    return result



def read_imagefile(file) -> Image.Image:
    print(type(file))
    image = Image.open(BytesIO(file)).convert('RGB')
    return image