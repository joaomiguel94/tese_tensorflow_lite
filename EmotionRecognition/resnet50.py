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
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from keras.applications import ResNet50
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.optimizers import SGD, Adamax
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, Flatten, MaxPooling2D, InputLayer
from keras.layers import Dense
from keras.constraints import maxnorm
from keras import backend as K
#from skimage.feature import local_binary_pattern

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions

def get_files(emotion):
 
    files = glob.glob("data_set_150x150/%s/*" %emotion)
    random.shuffle(files)
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

random.shuffle(temp)

""" training = temp[:int(len(temp)*0.7)] #get first 80% of file list
test = temp[-int(len(temp)*0.1):] #get last 20% of file list
val = temp[-int(len(temp)*0.3):] #get last 20% of file list """

training, test, val  = np.split(temp, [int(.8 * len(temp)), int(.9 * len(temp))]) 
#random.shuffle(training)
#random.shuffle(test)
#random.shuffle(val)

""" for i in range(len(training)):
    print(training[i]) """

training_data = []
training_labels = []
test_data = []
test_labels = []
val_data = []
val_labels = []


def make_sets(training, test, val):
    '''This function creates test/train data and labels.'''
 
    #for emotion in emotions:
    #    print(" working on %s" %emotion)
        #training, prediction = get_files(emotion)

        #Append data to training and prediction list, and generate labels 0-7
    for item in training:      
        image = cv2.imread(item) #open image
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        img = img_to_array(image)
        #radius = 3
        #no_points = 8 * radius
        #lbp = local_binary_pattern(img, no_points, radius, method='uniform')
        #img = img/255              
        training_data.append(img) 
        training_labels.append(emotions.index(item.split("/")[1]))        
        #training_labels.append(emotions.index(emotion))
        #print(emotions.index(item.split("/")[1]))
            
    for item in test:
        image = cv2.imread(item)
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = img_to_array(image)
        #radius = 3
        #no_points = 8 * radius
        #lbp = local_binary_pattern(lbp, no_points, radius, method='uniform')
        #img = img/255              
        test_data.append(img)            
        test_labels.append(emotions.index(item.split("/")[1]))
        #test_labels.append(emotions.index(emotion))

    for item in val:
        image = cv2.imread(item)
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = img_to_array(image)
        #radius = 3
        #no_points = 8 * radius
        #lbp = local_binary_pattern(img, no_points, radius, method='uniform')
        #img = img/255              
        val_data.append(img)            
        val_labels.append(emotions.index(item.split("/")[1]))        
        #test_labels.append(emotions.index(emotion))

    return training_data, training_labels, test_data, test_labels, val_data, val_labels

batch_size = 64
nb_classes = 8
nb_epoch = 100

nb_filters = 3
pool_size = (2, 2)
kernel_size = (5, 5)

img_rows, img_cols = 150, 150

X_train2, y_train2, X_test2, y_test2, X_val2, y_val2 = make_sets(training, test, val)

X_train = np.array(X_train2)
y_train = np.array(y_train2)
X_test = np.array(X_test2)
y_test = np.array(y_test2)
X_val = np.array(X_val2)
y_val = np.array(y_val2)

X_train = X_train.astype("float32")
y_train = y_train.astype("float32")
X_test = X_test.astype("float32")
y_test = y_test.astype("float32")
X_val = X_val.astype("float32")
y_val = y_val.astype("float32")

class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
#print("class_weights 1:", class_weights)
class_weights = dict(enumerate(class_weights))
#print("class_weights 2:", class_weights)


print("Tamanho X_train:", X_train.shape)
print("Tamanho y_train:", y_train.shape)
print("Tamanho X_test:", X_test.shape)
print("Tamanho y_test:", y_test.shape)

print("Len X_train:", len(X_train))
print("Len y_train:", len(y_train))
print("Len X_test:", len(X_test))
print("Len y_test:", len(y_test))

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
Y_val = np_utils.to_categorical(y_val, nb_classes)

X_train = X_train.reshape(len(X_train), img_rows, img_cols, 3)
X_test = X_test.reshape(len(X_test), img_rows, img_cols, 3) 
X_val = X_val.reshape(len(X_val), img_rows, img_cols, 3) 

input_shape = (img_rows, img_cols, 3)
image_input = Input(shape=(img_rows, img_cols, 3))

print("Tamanho X_train:", X_train.shape)
print("Tamanho Y_train:", Y_train.shape)
print("Tamanho X_test:", X_test.shape)
print("Tamanho Y_test:", Y_test.shape)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#cv2.imwrite("X_test1.png", X_test[1])

#Substituir o modelo pelo ResNet50/101
#Testar com as imagens do novo dataset e juntar ao treino e JAFFE, KDEF

base_model = ResNet50(weights= None, include_top=False, input_shape= input_shape)
x = base_model.output
x = Flatten(name='flatten')(x)
predictions = Dense(8, activation='softmax', name='predictions')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[0:3]:
    layer.trainable = False

adam = optimizers.Adam(lr=0.0001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
start_time = datetime.now() 

history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_val, Y_val))
score = model.evaluate(X_test, Y_test)


time_elapsed = datetime.now() - start_time

print('Tempo de treino: ', time_elapsed)
print('Test score: ', score[0])
print('Test accuracy: ', score[1]*100)
print('Score: ', score)

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

""" # Plot the confusion matrix 
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
plt.show() """

model.save("model_resnet50.h5")