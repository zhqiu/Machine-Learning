# -*- coding: utf-8 -*-
"""	Qiu Zihao's homework of ML
	Student ID: 141130077
	Naive Bayes
"""

import pickle
import numpy as np

# read data from file 
def readData():
	trainData = pickle.load(open('train_data.pkl', 'rb')).todense()
	trainTargets = pickle.load(open('train_targets.pkl','rb'))
	testData = pickle.load(open('test_data.pkl','rb')).todense()
	return trainData, trainTargets, testData

# calculate the prior probablity of train set
# return type is dict like this:
#{0: [127, 0.1948], ...} 0 means tpye, 127 means frequency in tarinTagets, 0.1948 means probablity
def calPriorProb(trainTargets):
	typeDict = {}
	for mark in trainTargets:
		if mark in typeDict:
			typeDict[mark][0] += 1
		else:
			typeDict[mark] = [1]
	sum = len(trainTargets)
	for item in typeDict:
		typeDict[item].append((typeDict[item][0]+1)/(sum+len(typeDict))) # Laplace smoothing
	return typeDict


# calculate the conditional probablity from every discrete attributes(2500)
# return type is ndarray
# size: 5000*cols
def calCondPorb(trainData, trainTargets, typeDict, cols):
	condProb = np.zeros((5000, cols), dtype=np.float64)  # build conditional probablity matrix
	discMat = trainData.getA()[:, 0:2500]           # build discrete matrix of trainData
	for i, vect in enumerate(discMat):
		label = trainTargets[i]      # label of this sample
		for j, attr in enumerate(vect):  
			if attr==0:
				condProb[2*j][label] += 1
			elif attr==1:
				condProb[2*j+1][label] += 1
	for label in range(cols):
		for i in range(5000):
			condProb[i][label] = (condProb[i][label]+1)/(typeDict[label][0]+2) # Laplace smoothing
	return condProb


# calculate mean and variance of continuous attributes(2500)
def calMeanAndVar(trainData, trainTargets, cols):
	contMat = trainData.getA()[:, 2500:5000] 
	meanMat = np.zeros((2500, cols), dtype=np.float64)
	nonZeroMat = np.zeros((2500, cols))
	varMat = np.zeros((2500, cols), dtype=np.float64)
	# calculate mean matrix
	for num, line in enumerate(contMat):
		for i, attr in enumerate(line):
			if attr!=0:
				nonZeroMat[i][trainTargets[num]] += 1
				meanMat[i][trainTargets[num]] += attr
	for line in range(2500):
		for label in range(cols):
			if nonZeroMat[line][label]!=0:
				meanMat[line][label] /= nonZeroMat[line][label]
	# calculate variance matrix
	for num, line in enumerate(contMat):
		for i, attr in enumerate(line):
			if attr!=0:
				varMat[i][trainTargets[num]] += np.square(meanMat[i][trainTargets[num]]-attr)
	for line in range(2500):
		for label in range(cols):
			if nonZeroMat[line][label]!=0:
				varMat[line][label] /= nonZeroMat[line][label]
	return meanMat, varMat

# calculate probablity of continuous attributes
# x: value; attr: attribute
def probOfContAttr(x, attr, meanMat, varMat, label):
	if varMat[attr][label]==0:
		return 1
	numerator = np.exp(-np.square((x-meanMat[attr][label]))/(2*varMat[attr][label]))
	denominator = np.sqrt(2*np.pi*varMat[attr][label])
	if denominator==0:
		return 1
	return numerator/denominator

# predict
# ouput to test_predictions.csv
def predict(testData, typeDict, condProb, meanMat, varMat, cols):
	outfile = open('test_predictions.csv', 'w')
	testMat = testData.getA()
	for vect in testMat:
		discVect = vect[0:2500]
		contVect = vect[2500:5000]
		resultPorb = np.zeros((cols, 1), dtype=np.float)  # probablity of all five kinds
		for label in range(cols):
			resultPorb[label] += np.log(typeDict[label][1])
			for discAttr in range(2500):
				if discVect[discAttr]==0:
					resultPorb[label] += np.log(condProb[2*discAttr][label])
				elif discVect[discAttr]==1:
					resultPorb[label] += np.log(condProb[2*discAttr+1][label])
#			for contAttr in range(2500):
#				if contVect[contAttr]!=0:
#					resultPorb[label] += np.log(probOfContAttr(contVect[contAttr], contAttr, meanMat, varMat, label)) 
#		print(np.argmax(resultPorb))
		outfile.write(str(np.argmax(resultPorb))+'\n')
	outfile.close()
				

trainData, trainTargets, testData = readData() 
typeDict = calPriorProb(trainTargets)
condProb = calCondPorb(trainData, trainTargets, typeDict, len(typeDict)) 
meanMat, varMat = calMeanAndVar(trainData, trainTargets, len(typeDict)) 
predict(testData, typeDict, condProb, meanMat, varMat, len(typeDict))
