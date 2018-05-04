# -*- coding: utf-8 -*-
"""	Qiu Zihao's homework of ML
	Student ID: 141130077
	Neural Network(using Keras and Theano)
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# read in data
def readInData(datafile, labelfile):
	dataMat = np.genfromtxt(datafile, delimiter=',', dtype=np.float)
	labelMat = np.genfromtxt(labelfile, dtype=np.int)
	return dataMat, labelMat

dataMat, labelMat = readInData('train_data.csv', 'train_targets.csv')

# change labelMat(n*1) into targetMat(n*10)
targetMat = np.zeros([labelMat.shape[0], 10])
i = 0
for res in labelMat:
	targetMat[i][res] = 1
	i += 1
	
# creat model
model = Sequential()
model.add(Dense(400, input_dim=dataMat.shape[1]))                      # input layer dim=400
model.add(Dense(512, activation='relu'))   # hidden layer 1
model.add(Dense(512, activation='relu'))   # hidden layer 2
model.add(Dense(10, activation='softmax')) # output layer

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam')

# fit the model
model.fit(dataMat, targetMat, epochs=10)

# get answer from the result vector
def getAnswer(result):  
	return np.argmax(result)

# start to predict
testDataMat = np.genfromtxt('test_data.csv', delimiter=',', dtype=np.float)
predictions = model.predict(testDataMat)
outfile = open('test_predictions_library.csv', 'w')
for result in predictions:
	outfile.write(str(getAnswer(result))+'\n')
outfile.close()
