# -*- coding: utf-8 -*-
"""	Qiu Zihao's homework of ML
	Student ID: 141130077
	Neural Network
"""

import numpy as np

# input layer -- 400 (d in book)
input_num = 400
# hidden layer -- 100 (q in book)
hidden_num = 100
# output layer -- 10 (l in book)
output_num = 10

# connection weights from input layer to hidden layer(v in book)
v = (np.random.random(size=(input_num, hidden_num))-0.5)/10

# connection weights from hidden layer to output layer(w in book)
w = (np.random.random(size=(hidden_num, output_num))-0.5)/10

# threshold of hidden layer(gama in book)
gama = np.zeros([1, hidden_num])

# threshold of output layer(theta in book)
theta = np.zeros([1, output_num])

# read in data
def readInData(datafile, labelfile):
	dataMat = np.genfromtxt(datafile, delimiter=',', dtype=np.float)
	labelMat = np.genfromtxt(labelfile, dtype=np.int)
	return dataMat, labelMat

# sigmoid function
def sigmoid(x):
	return 1.0/(1+np.exp(-x))

# calculate output 
def calOutput(indata):
	# input of the hidden layer unit 
	alpha = np.dot(v.T, indata)
	# output of the hidden layer unit
	b = sigmoid(alpha - gama) 
	# input of the output layer unit
	beta = np.dot(b, w) 
	# output of the output layer unit
	y = sigmoid(beta - theta)
	return b, y

# calculate the gradient item g(in book)
def calG(output ,label): 
	return output*(1-output)*(label-output)
	

# calculate the gradient item e(in book)
def calE(b, g): 
	return b*(1-b)*sum(np.dot(g, w.T))

dataMat, labelMat = readInData('train_data.csv', 'train_targets.csv') 

# return the i th label vector 
def labelVector(i):
	vec = np.zeros(10)
	vec[labelMat[i]] = 1
	return vec

# train the NN model
times = 16
step = 0.2
while (times > 0):
	if times<8:
		step = 0.1
	print(times)
	for i in range(dataMat.shape[0]):
		data = dataMat[i]
		b, y = calOutput(data)
		g = calG(y, labelVector(i))
		e = calE(b, g) 
		# renew the connection weights and threshold
		w += step*(np.dot(b.T, g)) 
		v += step*(np.dot(np.array([data]).T, e)) 
		gama += -step*e;
		theta += -step*g; 
	times -= 1

# get answer from the result vector
def getAnswer(result):  
	return np.argmax(result)

# start to test
testDataMat = np.genfromtxt('test_data.csv', delimiter=',', dtype=np.float)
outfile = open('test_predictions.csv', 'w')
for i in range(testDataMat.shape[0]): 
	no_use, result=calOutput(testDataMat[i]) 
	ans = getAnswer(result[0]) 
	outfile.write(str(ans)+'\n')
outfile.close()

