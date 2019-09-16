import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
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
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping

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
    #X_validation = X_validation.reshape(X_validation.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    #X_validation = X_validation.reshape(X_validation.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
#print(X_validation.shape[0], 'validation samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
#Y_validation = np_utils.to_categorical(y_validation, nb_classes)

data_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.02,
    height_shift_range=0.02,
    horizontal_flip=True)
data_generator.fit(X_train)

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


#{0:'neutral', 1:'anger', 2:'contempt', 3:'disgust', 4:'fear', 5:'happy', 6:'sadness', 7:'surprise'}
""" 
class_weights = {0: 0.000761615,
                1: 0.001290323,
                2: 0.005376344,
                3: 0.001451379,
                4: 0.002398082,
                5: 0.000970874,
                6: 0.002380952,
                7: 0.000956023
                } """

#training

""" history = model.fit_generator(data_generator.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train) // batch_size,
                    epochs=nb_epoch,
                    validation_data=(X_test, Y_test),
                    verbose=1,
                    shuffle=True) """

""" history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, 
                    validation_split=0.1, validation_data=(X_test, Y_test), shuffle=True) """

#score = model.evaluate(X_test, Y_test, verbose=0)

# K Fold
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

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
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


model.save("model.h5")