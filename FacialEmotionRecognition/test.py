import argparse
import numpy as np
import pickle
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
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt


if __name__ != '__main__':
    raise ImportError('Should be run as Script')

parser = argparse.ArgumentParser(
    description='''
        Convolutional Neural Netwok for training a facial emotion classifier.
    ''',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    epilog='''
        Examples:
            python %(prog)s /path/to/ck_dataset.pickle
    '''
)
parser.add_argument('dataset_path', help='Absolute Path of the pickled CK+ Dataset')

dataset_path = parser.parse_args().dataset_path



batch_size = 30
nb_classes = 8
nb_epoch = 50

# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (5, 5)

# the data, shuffled and split between train and test sets
with open(dataset_path, 'rb') as pickled_dataset:
    data_obj = pickle.load(pickled_dataset)

(training_data, validation_data, test_data) = data_obj['training_data'], data_obj['validation_data'], data_obj['test_data']
(X_train, y_train), (X_test, y_test) = (training_data[0],training_data[1]), (test_data[0],test_data[1])

# input image dimensions
img_rows, img_cols = data_obj['img_dim']['width'], data_obj['img_dim']['height']

#checks if backend is theano or tensorflow for dataset format
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

model = keras.models.load_model("model.h5")
model.summary()


predictions = model.predict(
    X_test,
    batch_size=batch_size
)
print(predictions.shape)

#print(pred)
emotions = ['neutral','anger','contempt','disgust','fear','happy', 'sadness', 'surprise']
""" 
print ("neutral: %", predictions[0]/1.0 * 100)
print ("anger: %", predictions[1]/1.0 * 100)
print ("contempt: %", predictions[2]/1.0 * 100)
print ("disgust: %", predictions[3]/1.0 * 100)
print ("fear: %", predictions[4]/1.0 * 100)
print ("happy: %", predictions[5]/1.0 * 100)	
print ("sadness: %", predictions[6]/1.0 * 100)
print ("happy: %", predictions[5]/1.0 * 100)	
print ("surprise: %", predictions[7]/1.0 * 100)	 """	
print ("----------------------"	)	

count = 0

for i in range(592):
    max_index = np.argmax(predictions[i])
    emotion = emotions[max_index]
    #true_id = y_test[i]
    print(f'predicted: {max_index} true value: {y_test[i]}')

    #if (i < 10):
    #    plt.imshow(X_test[i])

    if (max_index == y_test[i]):
        count+=1
print((count/592)*100)