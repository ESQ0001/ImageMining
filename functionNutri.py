# -*- coding: utf-8 -*-
"""
Created on Sat May 16 20:34:36 2020

@author: aecan
"""
import csv

import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,AveragePooling2D,Conv2D, MaxPooling2D, Flatten, Input, GlobalAveragePooling3D
from keras.applications import MobileNet, ResNet50, VGG16, InceptionV3
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint

import pandas as pd

def getNutritionalValues(predictionIndex):
    

df = pd.read_csv("C://Users/Desktop/nutritionalInfo_desserts.csv")
return df.loc[predictionIndex]

def load_model():
    json_file = open('C:/Users/aecan/Desktop/modelVGG16_66.json', 'r')
    loaded_model_json2 = json_file.read()
    json_file.close()
    loaded_model2 = model_from_json(loaded_model_json2)
    loaded_model2.load_weights("C:/Users/aecan/Desktop/modelWeightsVGG16_66.h5")
    opt = Adam(lr=0.00001)
    loaded_model2.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    model = loaded_model2
    df_whole = pd.read_csv("drive/My Drive/Colab Notebooks/whole_food_datasetColab.csv")

 def predict3(im,model):

        print ("I am entering predict3")
        pickle_in = open("C:/Users/aecan/Desktop/labels_desserts.pickle","rb")
        pixels = pickle.load(pickle_in)
        pickle_off = open("C:/Users/aecan/Desktop/pixels_desserts.pickle","rb")
        labels = pickle.load(pickle_off) 
        Xnew = np.array(im)
        Xnew = np.expand_dims(Xnew, axis =0)
        Xnew = imagenet_utils.preprocess_input(Xnew)
        y_prob = model.predict(Xnew)[0]
        print(model.predict_classes(Xnew))
        idxs = np.argsort(y_prob)[0]
        foods = np.unique(labels)
        #foods[model.predict_classes(Xnew)]
        return idxs
        #return 'churros'
def output(im,model):
    return getNutritionalValues(predict3(im,model))
