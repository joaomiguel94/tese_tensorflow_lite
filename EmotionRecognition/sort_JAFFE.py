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
from sklearn.utils import class_weight
import shutil

folders = glob.glob("/home/joao/Desktop/jaffedbase/jaffedbase/*")
folders.sort()
sub_folders = []

for j in range(len(folders)):
    print(folders[j])
    image = os.path.splitext(folders[j].split(".")[1])[0]
    print(image)
    a = image[0:0+2]
    print(a)
    print("-------------------------")
    print("-------------------------")


    if (a == 'FE'):
        shutil.copy2(folders[j], '/home/joao/Desktop/Tese/EmotionRecognition/JAFFE_Sorted/fear')

    if (a == 'AN'):
        shutil.copy2(folders[j], '/home/joao/Desktop/Tese/EmotionRecognition/JAFFE_Sorted/anger')

    if (a == 'DI'):
        shutil.copy2(folders[j], '/home/joao/Desktop/Tese/EmotionRecognition/JAFFE_Sorted/disgust')

    if (a == 'SA'):
        shutil.copy2(folders[j], '/home/joao/Desktop/Tese/EmotionRecognition/JAFFE_Sorted/sadness')

    if (a == 'HA'):
        shutil.copy2(folders[j], '/home/joao/Desktop/Tese/EmotionRecognition/JAFFE_Sorted/happy')

    if (a == 'NE'):
        shutil.copy2(folders[j], '/home/joao/Desktop/Tese/EmotionRecognition/JAFFE_Sorted/neutral')

    if (a == 'SU'):
        shutil.copy2(folders[j], '/home/joao/Desktop/Tese/EmotionRecognition/JAFFE_Sorted/surprise')
