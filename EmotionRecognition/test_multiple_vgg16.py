import cv2
import glob
import tensorflow as tf
from shutil import copyfile
import os
import pandas as pd
import csv
import random
import math
import numpy as np
#import dlib
import itertools
from sklearn.svm import SVC
import pickle
import argparse
import imghdr
from datetime import datetime 
import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, MaxPool2D
from keras.utils import np_utils
from keras import backend as K
from keras.layers import Activation, Convolution2D, Dropout, Conv2D, ZeroPadding2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2
from keras.optimizers import SGD,Adadelta
from sklearn.metrics import confusion_matrix
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import dlib
import matplotlib.pyplot as plt
from keras import optimizers
from keras.models import load_model
from keras.preprocessing import image
from keras import metrics


emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions

def get_files(emotion):
 
    files = glob.glob("imagens/%s/*" %emotion)
    #files = glob.glob("data_set_150x150/%s/*" %emotion)
    #random.shuffle(files)
    #training = files[:int(len(files)*0.8)] #get first 80% of file list
    #test = files[-int(len(files)*0.2):] #get last 20% of file list
    #return training, test
    return files

training = []
test = []
temp = []

for emotion in emotions:
    print(" working on %s" %emotion)

    taux = get_files(emotion)
    temp = temp + taux
    #taux, paux = get_files(emotion)

    #training = training + taux
    #test = test + paux

#random.shuffle(temp)

training, test, val  = np.split(temp, [int(.8 * len(temp)), int(.9 * len(temp))]) 
test = temp

""" random.shuffle(training)
random.shuffle(test)
random.shuffle(val) """
""" 
for i in range(len(training)):
    print(training[i]) """

training_data = []
training_labels = []
test_data = []
test_labels = []
val_data = []
val_labels = []


def make_sets():
    '''This function creates test/train data and labels.'''
 
    #for emotion in emotions:
    #    print(" working on %s" %emotion)
        #training, prediction = get_files(emotion)

        #Append data to training and prediction list, and generate labels 0-7
    for item in training:      
        image = cv2.imread(item) #open image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        img = img_to_array(gray)
        #img = img/255              
        training_data.append(img) 
        training_labels.append(emotions.index(item.split("/")[1]))        
        #training_labels.append(emotions.index(emotion))
        #print(emotions.index(item.split("/")[1]))
            
    for item in test:
        image = cv2.imread(item)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = img_to_array(gray)
        #img = img/255              
        test_data.append(img)            
        test_labels.append(emotions.index(item.split("/")[1]))        
        #test_labels.append(emotions.index(emotion))

    for item in val:
        image = cv2.imread(item)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = img_to_array(gray)
        #img = img/255              
        val_data.append(img)            
        val_labels.append(emotions.index(item.split("/")[1]))        
        #test_labels.append(emotions.index(emotion))

    return training_data, training_labels, test_data, test_labels, val_data, val_labels

batch_size = 64
nb_classes = 8
nb_epoch = 50

nb_filters = 3
pool_size = (2, 2)
kernel_size = (5, 5)

img_rows, img_cols = 150, 150

X_train2, y_train2, X_test2, y_test2, X_val2, y_val2 = make_sets()

#model = load_model('model_vgg16.h5')


""" dependencies = {
    'accuracy': keras.metrics.BinaryAccuracy,
    'precision': keras.metrics.Precision,
    'recall': keras.metrics.Recall,
    'auc': keras.metrics.AUC,
} """

#model = keras.models.load_model('model_weights.h5', custom_objects=dependencies)
#model = keras.models.load_model('model_weights.h5')
model = keras.models.load_model('model_lstm.h5')


""" metrics = [
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
] """

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

count = 0
emotion_ant = ""

array = [0, 0, 0, 0, 0, 0, 0, 0]

for i in range(len(test)):

    test_image = image.load_img(test[i], color_mode = "grayscale", target_size = (150, 150)) 
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    
    print(f"Imagem: {test[i]}")
    for j in range(len(result)):

        #print(f"{emotions[j]}: {result[j]*100}")
        print ("Neutral: %", str(round(result[0][0]/1.0 * 100, 4)))
        print ("Anger: %", str(round(result[0][1]/1.0 * 100, 4)))
        print ("Contempt: %", str(round(result[0][2]/1.0 * 100, 4)))
        print ("Disgust: %", str(round(result[0][3]/1.0 * 100, 4)))
        print ("Fear: %", str(round(result[0][4]/1.0 * 100, 4)))
        print ("Happy: %", str(round(result[0][5]/1.0 * 100, 4)))	
        print ("Sadness: %", str(round(result[0][6]/1.0 * 100, 4)))		
        print ("Surprise: %", str(round(result[0][7]/1.0 * 100, 4)))		
        max_index = np.argmax(result[0])
        emotion = emotions[max_index]
        print (f"Prediction: {emotion}")
        if max_index == emotions.index(test[i].split("/")[1]):
            count = count +1
            print(f"Veredicto: Acertou")
            array[max_index] = array[max_index] + 1
        else:
            print(f"Veredicto: Falhou")
        print ("----------------------"	)

print ("")              
print(f"Nº Imagens: {len(test)}")
print(f"Nº certadas: {count}")
print(f"Percentagem acertadas: {(count/len(test))*100}")
print ("")
print(f"Acerto por classe:")
for i in range(len(array)):
    print(f"{emotions[i]}: {array[i]}")