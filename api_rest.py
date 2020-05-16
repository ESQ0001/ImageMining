# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:10:44 2020

@author: aecan
"""

### In requires the following packages
### conda install -c conda-forge flask-restful
### conda install pillow
import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,AveragePooling2D,Conv2D, MaxPooling2D, Flatten, Input, GlobalAveragePooling3D
from keras.applications import MobileNet, ResNet50, VGG16, InceptionV3
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint

from PIL import Image
import base64
import io
import numpy as np

from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from PIL import Image
import numpy as np
import io
import os
#from google.colab.patches import cv2_imshow
#from sklearn.datasets import make_blobs
from skimage.io import imread
import pickle
from keras.models import model_from_json

# if you have used sklearn for generating the model
import joblib
import torch
app = Flask(__name__) # Server
api = Api(app) # api-rest

#  Loading the pre-trained model
# load json and create model 

json_file = open('C:/Users/aecan/Desktop/modelVGG16_66.json', 'r')
loaded_model_json2 = json_file.read()
json_file.close()
loaded_model2 = model_from_json(loaded_model_json2)
loaded_model2.load_weights("C:/Users/aecan/Desktop/modelWeightsVGG16_66.h5")
opt = Adam(lr=0.00001)
loaded_model2.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
model = loaded_model2



#model = joblib.load('svm_model_mnist_pixels.joblib') 

def feature_extraction(image):
    # preprocessing
    im = image.astype(np.float32) / 255.
    # TODO: real feature extraction
    im = np.reshape(im, (1, np.prod(im.shape))) # we need a vector [1 x num_features] (depends on classifier)
    features = im
    return features

# We create a resource for the api
class Prediction(Resource):
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
        #return foods[model.predict_classes(Xnew)]
        return 'churros'

    
    
    @staticmethod
    def post():
        data = {"success": False}
    
        if request.get_json != None:
            test_image_base64_encoded = request.get_json().get('image')
            print(type(test_image_base64_encoded))
            base64_decoded = base64.b64decode((test_image_base64_encoded))
            print("HERE")
            image_np = np.array(base64_decoded)
            image_torch = torch.tensor(np.array(image_np))
            pred = predict3(model,image_torch)
            data['prediction'] = pred
            data['success'] = True # Indicate that the request was a sucess
            return jsonify(data) # Response

api.add_resource(Prediction, '/predict')

if __name__ == "__main__":
    print('Loading model and Flask starting server...')
    print('please wait until server has fully started')

    app.run(debug=True, use_reloader= False, host='0.0.0.0', port=5000) # Debug mode and open to all connections in port 5000