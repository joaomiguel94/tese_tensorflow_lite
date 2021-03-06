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
    files = glob.glob("data_set/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction


""" def make_sets():
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
    return training_data, training_labels, prediction_data, prediction_labels """

training = []
prediction = []

for emotion in emotions:
    print(" working on %s" %emotion)
    training, prediction = get_files(emotion)

for x in range(len(training)): 
    print( training[x])
print("-----------------")    
for x in range(len(prediction)): 
    print( prediction[x])

batch_size = 30
nb_classes = 8
nb_epoch = 250

# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (5, 5)

# input image dimensions
img_rows, img_cols = 350, 350

def image_generator(set_of_data, bs, mode="train", aug=None):

    while True:

        images = []
        labels = []

        while len(images) < bs:

            for item in set_of_data:
                image = cv2.imread(item) #open image
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
                img = img_to_array(gray)
                img = img/255              
                images.append(img) 
                labels.append(emotions.index(emotion))
        
        Imgs = np.array(images)
        Imgs = Imgs.astype("float32")
        Lbs = np.array(labels)
        Lbs = Lbs.astype("float32")

        Lbs2 = np_utils.to_categorical(Lbs, nb_classes)  
        Imgs2 = Imgs.reshape(len(Imgs), img_rows, img_cols, 1)

        if aug is not None:
            (Imgs2, Lbs2) = next(aug.flow(np.array(Imgs2),
            Lbs2, batch_size=bs))
      
        yield (np.array(Imgs2), Lbs2)

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

trainGen = image_generator(training, batch_size, mode="train", aug=aug)
testGen = image_generator(prediction, batch_size, mode="train", aug=None)

""" def generator(make_sets()):#X_data, y_data, batch_size):

  samples_per_epoch = X_data.shape[0]
  number_of_batches = samples_per_epoch/batch_size
  counter=0

  while 1:

    X_train2, y_train2, X_test2, y_test2 = make_sets()

    X_batch = np.array(X_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
    y_batch = np.array(y_data[batch_size*counter:batch_size*(counter+1)]).astype('float32')
    counter += 1
    yield X_batch,y_batch

    #restart counter to yeild data in the next epoch as well
    if counter >= number_of_batches:
        counter = 0 """



""" #X_train2, y_train2, X_test2, y_test2 = make_sets()

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
print("Len y_test:", len(y_test)) """

""" 
data_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.02,
    height_shift_range=0.02,
    horizontal_flip=True) 
data_generator.fit(X_train)"""

""" data_generator = ImageDataGenerator(
    rotation_range=20, 
    zoom_range=0.15,
	width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.15,
	horizontal_flip=True, 
    fill_mode="nearest")
data_generator.fit(X_train) """


""" def generator(features, labels, batch_size): # Create empty arrays to contain batch of features and labels# 
    
    batch_features = np.zeros((batch_size, img_cols, img_rows, 1))
    batch_labels = np.zeros((batch_size,1)) 
    while True:
        for i in range(batch_size):
            # choose random index in features
            index= random.choice(len(features),1)
            batch_features[i] = some_processing(features[index])
            batch_labels[i] = labels[index]
    yield batch_features, batch_labels """


#Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)

#X_train = X_train.reshape(len(X_train), img_rows, img_cols, 1)
#X_test = X_test.reshape(len(X_test), img_rows, img_cols, 1) 

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

""" model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
"""

""" history = model.fit_generator(data_generator.flow(X_train, Y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=nb_epoch,
        validation_data=(X_test, Y_test),
        verbose=1,
        shuffle=True)  """

""" history = model.fit_generator(data_generator.flow(X_train, Y_train, batch_size=batch_size),
	validation_data=(X_test, Y_test), steps_per_epoch=len(X_train) // batch_size,
	epochs=nb_epoch) """

#history = model.fit_generator(generator(X_train, Y_train, batch_size), samples_per_epoch=50, nb_epoch=nb_epoch)
#score = model.evaluate(X_test, y_test, verbose=0)

history = model.fit_generator(
	trainGen,
	steps_per_epoch=len(training) // batch_size,
	validation_data=testGen,
	validation_steps=30,
	epochs=nb_epoch)

time_elapsed = datetime.now() - start_time 

#print('Test score: ', score[0])
#print('Test accuracy: ', score[1]*100)
#print('Total training time: ', format(time_elapsed))

""" # K Fold
for i in range(5):
    print("Training on Fold: ",i+1)
    t_x, val_x, t_y, val_y = train_test_split(X_train, Y_train, test_size=0.1, random_state = np.random.randint(1,1000, 1)[0], shuffle=True)
    history = model.fit_generator(data_generator.flow(t_x, t_y, batch_size=batch_size), steps_per_epoch=len(X_train) // batch_size, epochs=nb_epoch, 
                                                      verbose=1, validation_data=(val_x, val_y), shuffle=True)
    score = model.evaluate(val_x, val_y, verbose=0)

    print("======="*12, end="\n\n\n")

time_elapsed = datetime.now() - start_time

print('Test score: ', score[0])
print('Test accuracy: ', score[1]*100)
print('Total training time: ', format(time_elapsed))
 """

""" # Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot the confusion matrix 
y_pred = model.predict_classes(X_train)
y_true = [0] * len(y_pred)

for i in range(0, len(Y_train)):
	max_index = np.argmax(Y_train[i])
	y_true[i] = max_index

cm = confusion_matrix(y_pred, y_true, labels=range(nb_classes))

#{0:'neutral', 1:'anger', 2:'contempt', 3:'disgust', 4:'fear', 5:'happy', 6:'sadness', 7:'surprise'}

labels = ['neutral' , 'anger' , 'contempt' , 'disgust'  , 'fear' ,  'happy' , 'sadness', 'surprise']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

 """
model.save("model.h5")