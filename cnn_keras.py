'''
cnn_keras.py by Dominic Reichl, @domreichl
a convolutional neural network to categorize handwritten digits using Keras
'''

import keras, numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers.convolutional import Conv2D # for convolutional layers
from keras.layers.convolutional import MaxPooling2D # for pooling layers
from keras.layers import Flatten # to flatten data for fully connected layers

print('\n\n\nConvolutional NN with Keras')
print('-'*30)

''' 1. load data '''

data = numpy.load('mnist.npz') # load data (download from 'https://s3.amazonaws.com/img-datasets/mnist.npz')
X_train = data['x_train'] # training features
X_test = data['x_test'] # test features
y_train = data['y_train'] # training labels
y_test = data['y_test'] # test labels

''' 2. preprocess data '''

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

X_train = X_train / 255 # normalize training data
X_test = X_test / 255 # normalize test data

y_train = to_categorical(y_train) # encode training outputs
y_test = to_categorical(y_test) # encode test outputs

numClasses = y_test.shape[1] # number of categories

''' 3. build neural network '''

def convolutional_model():
    model = Sequential() # create model
    model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu', input_shape=(28, 28, 1))) # convolutional layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) # pooling layer
    
    model.add(Flatten()) # flatten for fully connected layer
    model.add(Dense(100, activation='relu')) # fully connected layer
    model.add(Dense(numClasses, activation='softmax')) # output layer
    
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    return model

''' 4. build, fit, and evaluate the model '''

model = convolutional_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

score = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: {} \n Error: {}".format(score[1], 100-score[1]*100))
