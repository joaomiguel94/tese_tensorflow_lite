import argparse
import numpy as np
import pickle

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

from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

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
(X_train, y_train), (X_test, y_test), (X_validation, y_validation) = (training_data[0],training_data[1]), (test_data[0],test_data[1]), (validation_data[0],validation_data[1])

# input image dimensions
img_rows, img_cols = data_obj['img_dim']['width'], data_obj['img_dim']['height']

#checks if backend is theano or tensorflow for dataset format
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_validation = X_validation.reshape(X_validation.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_validation = X_validation.reshape(X_validation.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_validation.shape[0], 'validation samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
Y_validation = np_utils.to_categorical(y_validation, nb_classes)


#---------------------------------------------------------------
 #A sequential model (feedforward)
model = Sequential()

#adding 2 Convolutional Layers and a maxpooling layer with activation function rectified linear unit and  Dropout for regularization
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size, strides=1))
model.add(Dropout(0.25))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size, strides=1))
model.add(Dropout(0.25))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size, strides=1))
model.add(Dropout(0.25))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size, strides=1))
model.add(Dropout(0.25))

#A Fully Conntected Layer with relu and a output layer with softmax
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

#-----------------------------------------------------------------
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='categorical_hinge', optimizer='rmsprop',metrics=["accuracy"])

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
#training

for i in range(5):
    print("Training on Fold: ",i+1)
    t_x, val_x, t_y, val_y = train_test_split(X_train, Y_train, test_size=0.1, random_state = np.random.randint(1,1000, 1)[0], shuffle=True)
    model.fit(t_x, t_y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(val_x, val_y), shuffle=True,
    class_weight=class_weights)
    score = model.evaluate(val_x, val_y, verbose=0)

    print("======="*12, end="\n\n\n")

print('Test score:', score[0])
print('Test accuracy:', score[1]*100)

model.save("model.h5")