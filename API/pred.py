#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 00:56:07 2021

@author: elilsonsantos
"""
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import  Image
#from tensorflow.keras.applications import decode_predictions

input_shape = (224,224)

def load_model():

    model = tf.keras.models.load_model('model/my_model.hdf5')
    
    return model

_model = load_model()


def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded))
    
    return pil_image

def preprocess(image: Image.Image):
    image = image.resize(input_shape)
    image = np.asfarray(image)
    image = image / 127.5 - 1.0
    image = np.expand_dims(image, 0)
    
    return image

def predict(image: np.ndarray):
    predictions = _model.predict(image)
    print(predictions)
    #predictions = deecode_preditions(_model.predict[0][0][1])
    return predictions