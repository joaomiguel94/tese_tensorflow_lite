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

files = []
def get_files(emotion):
    '''Define function to get file list, randomly shuffle it 
    and split 80/20'''
    files = glob.glob("data_set4/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    test = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, test
    #return files

training = []
test = []
temp = []

for emotion in emotions:
    print(" working on %s" %emotion)

    #taux = get_files(emotion)
    #temp = temp + taux
    taux, paux = get_files(emotion)

    training = training + taux
    test = test + paux

""" random.shuffle(temp)
random.shuffle(temp)

training = temp[:int(len(temp)*0.8)] #get first 80% of file list
test = temp[-int(len(temp)*0.2):] #get last 20% of file list """

random.shuffle(training)
random.shuffle(test)

for x in range(len(training)): 
    print(f'{training[x]}') 

print("------------------------------------------------------------------------")    
print("------------------------------------------------------------------------")    
print("------------------------------------------------------------------------")    

for x in range(len(test)): 
    print(f'{test[x]}')

print("------------------------------------------------------------------------")
print("")
print(f'Imagens de treino: {len(training)}') 
print(f'Imagens de teste: {len(test)}')
print(f'Total: {len(test) + len(training)}')
print("")

batch_size = 64
nb_classes = 8
nb_epoch = 50

nb_filters = 64
pool_size = (2, 2)
kernel_size = (5, 5)
img_rows, img_cols = 350, 350

def image_generator(set_of_data, bs):

    images = []
    labels = []
    
    while True:

        for item in set_of_data:

            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            img = img_to_array(gray)
            #img = img/255              
            images.append(img) 
            labels.append(emotions.index(emotion))
            
            Imgs = np.array(images)
            Imgs = Imgs.astype("float32")
            Lbs = np.array(labels)
            Lbs = Lbs.astype("float32")

            Lbs2 = np_utils.to_categorical(Lbs, nb_classes)  
            Imgs2 = Imgs.reshape(len(Imgs), img_rows, img_cols, 1)

            if len(images) == bs:
            
                yield (Imgs2, Lbs2)

                images.clear()
                labels.clear()

""" def image_generator(set_of_data, bs, mode="train", aug=None):

    images = []
    labels = []
    
    while True:

        images.clear()
        labels.clear()

        while len(images) < bs:

            for item in set_of_data:
                image = cv2.imread(item)
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

        #print(f'Length images: {len(images)}')

        #if aug is not None:
        #    (Imgs2, Lbs2) = next(aug.flow(Imgs2, Lbs2, batch_size=bs))
      
        yield (Imgs2, Lbs2) """

""" aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest") """

trainGen = image_generator(training, batch_size)
testGen = image_generator(test, batch_size)

#Model
input_shape = (img_rows, img_cols, 1)
model = Sequential()

model.add(Conv2D(filters=8, kernel_size=(7, 7), padding='same', name='image_array', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(filters=8, kernel_size=(7, 7), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=2,  padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(filters=nb_classes, kernel_size=(3, 3), padding='same'))
model.add(GlobalAveragePooling2D())
model.add(Activation('softmax',name='predictions'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

start_time = datetime.now()  

#print('steps_per_epoch: ', len(training) // batch_size)

#training
model.fit_generator(trainGen,	
    steps_per_epoch=len(training) // batch_size, 
    validation_data=testGen,
	validation_steps=len(test) // batch_size,
	epochs=nb_epoch)

time_elapsed = datetime.now() - start_time 

#print('Test score: ', score[0])
#print('Test accuracy: ', score[1]*100)
#print('Total training time: ', format(time_elapsed))

score = model.evaluate(training, test, verbose=0)

time_elapsed = datetime.now() - start_time 

print('Test score: ', score[0])
print('Test accuracy: ', score[1]*100)
print('Total training time: ', format(time_elapsed))

""" 
# Plot training & validation accuracy values
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
model.save("model_final.h5")
