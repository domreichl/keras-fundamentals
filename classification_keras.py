'''
classification_keras.py by Dominic Reichl, @domreichl
a classification model to categorize handwritten digits using Keras
'''

import keras, numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt

print('\n\n\nClassification with Keras')
print('-'*30)

''' 1. load and inspect data '''

data = numpy.load('mnist.npz') # load data (download from 'https://s3.amazonaws.com/img-datasets/mnist.npz')
X_train = data['x_train'] # training features
X_test = data['x_test'] # test features
y_train = data['y_train'] # training labels
y_test = data['y_test'] # test labels

print(X_train.shape) # data structure
plt.imshow(X_train[0]); plt.show() # plot first training item

''' 2. preprocess data '''

numPx = X_train.shape[1] * X_train.shape[2] # get size of 1d vector
X_train = X_train.reshape(X_train.shape[0], numPx).astype('float32') # flatten training images
X_test = X_test.reshape(X_test.shape[0], numPx).astype('float32') # flatten test images
                          
X_train = X_train / 255 # normalize training inputs
X_test = X_test / 255 # normalize test inputs

y_train = to_categorical(y_train) # encode training outputs
y_test = to_categorical(y_test) # encode test outputs

numClasses = y_test.shape[1] # get num of outputs
print(numClasses)

''' 3. build neural network '''

def classification_model():
    model = Sequential() # create model
    model.add(Dense(numPx, activation='relu', input_shape=(numPx,))) # input layer
    model.add(Dense(100, activation='relu')) # hidden layer
    model.add(Dense(numClasses, activation='softmax')) # output layer

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # compile model
    return model

''' 4. train, test, and evaluate the network '''

model = classification_model() # build the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, verbose=2) # fit the model

score = model.evaluate(X_test, y_test, verbose=0) # evaluate the model
print('Accuracy: {}% \n Error: {}'.format(score[1], 1 - score[1])) # display model performance
model.save('classification_model.h5') # save the model for future use
