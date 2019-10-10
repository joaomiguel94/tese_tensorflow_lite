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


#faceDet = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
#faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
#faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
 
 
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#detector = dlib.get_frontal_face_detector()
 
#predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
 
#clf = SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel

#data = {} #Make dictionary for all values
#data['landmarks_vectorised'] = []
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions

""" def detect_faces(emotion):
 
    #emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"] #Emotion list
     
    files = glob.glob("sorted_set/%s/*" %emotion) #Get list of all images with emotion
    filenumber = 0
 
    for f in files:
        frame = cv2.imread(f) #Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    
        #Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, 
        scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), 
        flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, 
        scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), 
        flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, 
        scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), 
        flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, 
        scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), 
        flags=cv2.CASCADE_SCALE_IMAGE)
 
        #Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        elif len(face_two) == 1:
           facefeatures = face_two
        elif len(face_three) == 1:
            facefeatures = face_three
        elif len(face_four) == 1:
            facefeatures = face_four
        else:
            facefeatures = ""
         
        #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            print("face found in file: %s" %f)
            gray = gray[y:y+h, x:x+w] #Cut the frame to size
            try:
                out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                cv2.imwrite("sorted_set/%s/%s.jpg" %(emotion, filenumber), out) #Write image
            except:
                pass #If error, pass file
        filenumber += 1 #Increment image number
 
for emotion in emotions:
     detect_faces(emotion) """

def get_files(emotion):
    '''Define function to get file list, randomly shuffle it 
    and split 80/20'''
 
    files = glob.glob("data_set/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction 

""" def get_landmarks(image):
    '''This function locates facial landmarks and computes 
    the relative distance from the mean for each point.'''
 
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, 
        ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, 
            x)*360)/(2*math.pi))
        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"
 """
def make_sets():
    '''This function creates test/train data and labels.'''
 
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        print(" working on %s" %emotion)
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            #convert to grayscale
            #out = cv2.resize(gray, (350, 350))
            #clahe_image = clahe.apply(out)
            #get_landmarks(clahe_image)
            #if data['landmarks_vectorised'] == "error":
            #    print("no face detected on this one")
            #else:
            img = img_to_array(gray)
            img = img/255              
            training_data.append(img) 
            training_labels.append(emotions.index(emotion))
                
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #out = cv2.resize(gray, (350, 350))
            #clahe_image = clahe.apply(out)
            #get_landmarks(clahe_image)
            #if data['landmarks_vectorised'] == "error":
            #    print("no face detected on this one")
            #else:
            img = img_to_array(gray)
            img = img/255              
            prediction_data.append(img)                    
            prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels,prediction_data, prediction_labels

batch_size = 30
nb_classes = 8
nb_epoch = 5

# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (5, 5)

X_train2, y_train2, X_test2, y_test2 = make_sets()

X_train = np.array(X_train2)
#print("Tamanho X_train:", X_train.shape)
y_train = np.array(y_train2)
X_test = np.array(X_test2)
y_test = np.array(y_test2)

X_train = X_train.astype("float32")
y_train = y_train.astype("float32")
X_test = X_test.astype("float32")
y_test = y_test.astype("float32")



print("Tamanho X_train:", X_train.shape)
print("Tamanho y_train:", y_train.shape)
print("Tamanho X_test:", X_test.shape)
print("Tamanho y_test:", y_test.shape)

print("Len X_train:", len(X_train))
print("Len y_train:", len(y_train))
print("Len X_test:", len(X_test))
print("Len y_test:", len(y_test))


# input image dimensions
img_rows, img_cols = 350, 350
 
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.reshape(len(X_train), img_rows, img_cols, 1)
X_test = X_test.reshape(len(X_test), img_rows, img_cols, 1)



input_shape = (img_rows, img_cols, 1)

#print("Tamanho X_train:", X_train.shape)
#print("Tamanho y_train:", y_train.shape)
#print("Tamanho X_test:", X_test.shape)
#print("Tamanho y_test:", y_test.shape)


#print('X_train shape:', X_train.shape)
#print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')

#print("Tamanho Y_train:", Y_train.shape)
#print("Tamanho Y_test:", Y_test.shape)


start_time = datetime.now()  

#early_stop = [EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')]
#---------------------------------------------------------------
 #A sequential model (feedforward)
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(7, 7), padding='same', name='image_array', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(filters=16, kernel_size=(7, 7), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=1,  padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=1, padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=nb_classes, kernel_size=(3, 3), padding='same'))
model.add(GlobalAveragePooling2D())
model.add(Activation('softmax',name='predictions'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#training

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
          
score = model.evaluate(X_test, Y_test, verbose=0)

time_elapsed = datetime.now() - start_time 

print('Test score: ', score[0])
print('Test accuracy: ', score[1]*100)
print('Total training time: ', format(time_elapsed))