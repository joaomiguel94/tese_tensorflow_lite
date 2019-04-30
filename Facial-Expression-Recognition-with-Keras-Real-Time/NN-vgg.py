import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
import scipy.misc
import dlib
import json
import cv2
from sklearn import preprocessing
from imutils import face_utils
import numpy as np
from keras.optimizers import SGD, Adadelta

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from numpy.random import seed
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

num_classes = 7 #'Angry','Disgust', 'Fear', 'Happy', 'Neutral', Sad', 'Surprise'
batch_size  = 32
epochs      = 300
fit         = True
seed(8)

# --------------------------------------------------------------------
# Data preperation from csv file
def prepare_data(fileName):
	X, Y = [], []
	x_train, y_train, x_test, y_test = [], [], [], []
	with open(fileName) as f:
		content = f.readlines()
		lines = np.array(content)
		num_of_instances = lines.size
		
		for i in range(1,num_of_instances):   	
			emotion, distances = lines[i].split(",")
			distances = distances.strip() 
			val = distances.split(" ")
			val = np.array(val)
			val = val.astype(np.float)

			emotion = keras.utils.to_categorical(emotion, num_classes)

			X.append(val)
			Y.append(emotion)
				
		x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=4)

		print ("********Training set size: ", str(len(x_train)))

		x_train = np.array(x_train)
		y_train = np.array(y_train)
		
		
		minmax = preprocessing.MinMaxScaler()
		x_train = minmax.fit_transform(x_train)
		
		x_test = np.array(x_test)
		y_test = np.array(y_test)

		# Normalization of the testing data   
		minmax = preprocessing.MinMaxScaler()
		x_test = minmax.fit_transform(x_test)

		return x_train, y_train, x_test, y_test

# --------------------------------------------------------------------
# Training and test data preperation  
x_train, y_train, x_test, y_test = prepare_data("data/training_CK.csv")

x_train = np.resize(x_train,(len(x_train),68,68,1))
x_test  = np.resize(x_test,(len(x_test),68,68,1))

print('Input images size: ', x_train.shape)
print('output labels size: ', y_train.shape)

print ("Training set size: ", str(len(x_train)))
print ("Test set size: ", str(len(x_test)))
# --------------------------------------------------------------------
# Construct the NN structure

img_rows, img_cols = 68, 68 
model = Sequential()
model.add(Conv2D(64, 5, 5, border_mode='valid',
                        input_shape=(img_rows, img_cols, 1)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(ZeroPadding2D(padding=(2, 2), dim_ordering='tf'))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(Conv2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(Conv2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(Conv2D(128, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(Conv2D(128, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

# compile the model

opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Train the model

history = model.fit(x_train,                            # Features
                      y_train,                          # Target
                      epochs=epochs,                    # Number of epochs
                      verbose=1,                        # No output
                      batch_size=batch_size,            # Number of observations per batch
                      validation_data=(x_test, y_test)) # Data for evaluation

# --------------------------------------------------------------------
# Visualize the training and test loss through epochs

# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

y_pred = model.predict_classes(x_test)
y_true = [0] * len(y_pred)

for i in range(0, len(y_test)):
	max_index = np.argmax(y_test[i])
	y_true[i] = max_index

# --------------------------------------------------------------------
# Print wrong classifications 

#for i in range(len(y_pred)):
#	if(y_pred[i] != y_true[i]):
#		print(str(i) + ' --> Predicted: ' +  str(y_pred[i]) + " Expected: " + str(y_true[i]))

# --------------------------------------------------------------------
# Draw the confusion matrix 
cm = confusion_matrix(y_pred, y_true, labels=range(num_classes))

labels = ['Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']
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

# --------------------------------------------------------------------
# Evaluate the model on the test set
scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# --------------------------------------------------------------------
# Save the model and the weights 
model_json = model.to_json()
model_json = model.to_json()
with open("model/model_in_json.json", "w") as json_file:
    json.dump(model_json, json_file)
model.save_weights("model/model333.h5")
print("Saved model to disk")
