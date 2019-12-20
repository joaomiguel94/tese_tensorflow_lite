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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
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
from keras.callbacks import EarlyStopping
import dlib
import matplotlib.pyplot as plt

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions

def get_files(emotion):
    '''Define function to get file list, randomly shuffle it 
    and split 80/20'''
    files = glob.glob("data_set4/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

training = []
prediction = []

for emotion in emotions:
    print(" working on %s" %emotion)
    taux, paux = get_files(emotion)
    #get_files(emotion, training, prediction)
    training = training + taux
    prediction = prediction + paux

for x in range(len(training)): 
    print(f'{training[x]}')

print("------------------------------------------------------------------------")    
print("------------------------------------------------------------------------")    
print("------------------------------------------------------------------------")    

for x in range(len(prediction)): 
    print(f'{prediction[x]}')

print("------------------------------------------------------------------------")    
print(len(prediction) + len(training))

batch_size = 30
nb_classes = 8
nb_epoch = 250

nb_filters = 64
pool_size = (2, 2)
kernel_size = (5, 5)
img_rows, img_cols = 150, 150

count2 = 0

def image_generator(set_of_data, bs, mode="train", aug=None):

    images = []
    labels = []
    count2 = 0

    #while True:

    for item in set_of_data:

        image = cv2.imread(item)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        img = img_to_array(gray)
        img = img/255              
        images.append(img) 
        labels.append(emotions.index(emotion))
        
        count2 = count2+1

        cv2.imwrite(set_of_data[count2],img)

        print(f'Name: {set_of_data[count2]}')
        print(f'Imagens: {len(images)}')

        print(f'Count: {count2}')

        if len(images) == bs:
        
            yield (images)

            images.clear()
            labels.clear()
            count2 = 0

image_generator(training, batch_size, mode="train", aug=None)

next(image_generator(training, batch_size, mode="train", aug=None))