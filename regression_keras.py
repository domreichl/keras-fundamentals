'''
regression_keras.py by Dominic Reichl, @domreichl
a regression model to predict concrete strength using Keras
'''

import keras, pandas
from keras.models import Sequential
from keras.layers import Dense

print('Regression with Keras')
print('-'*30)

''' 1. load and inspect data '''

regressionData = pandas.read_csv('https://ibm.box.com/shared/static/svl8tu7cmod6tizo6rk0ke4sbuhtpdfx.csv') # import data

print(regressionData.head()) # data structure
print(regressionData.shape) # num of rows & cols
print(regressionData.describe()) # data statistics
print(regressionData.isnull().sum()) # num of missing values

''' 2. preprocess data '''

regressionDataCols = regressionData.columns # all columns
predictors = regressionData[regressionDataCols[regressionDataCols != 'Strength']] # all columns except Strength
target = regressionData['Strength'] # Strength column
print(predictors.head()); print(target.head()) # sanity check

predictorsNorm = (predictors - predictors.mean()) / predictors.std() # normalize prediction data
numCols = predictorsNorm.shape[1] # num of predictors

''' 3. build neural network '''

def regression_model():
    model = Sequential() # create model
    model.add(Dense(50, activation='relu', input_shape=(numCols,))) # input layer
    model.add(Dense(50, activation='relu')) # hidden layer
    model.add(Dense(1)) # output layer
    
    model.compile(optimizer='adam', loss='mean_squared_error') # compile model
    return model

''' 4. train and test the network '''

model = regression_model() # build the model
model.fit(predictorsNorm, target, validation_split=0.3, epochs=100, verbose=2) # fit the model
