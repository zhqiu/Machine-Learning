# -*- coding: utf-8 -*-
"""
	Qiu Zihao's ML homework
	Student ID: 141130077
	AdaBoost
"""

import os
import warnings
from time import time
import numpy as np
from sklearn.model_selection import KFold
from sklearn import linear_model

# read in data
def readInData(datafile, labelfile):
	dataMat = np.genfromtxt(datafile, delimiter=',', dtype=np.float)
	labelMat = np.genfromtxt(labelfile, delimiter=',', dtype=np.float)
	return dataMat, labelMat 

# calculate error rate
def errorRate(D, regr, dataMat, labelMat):
	e = 0
	for i in range(len(dataMat)):
		if regr.predict(dataMat[i].reshape(1, -1)) != labelMat[i]:
			e += D[i]
	return e

# calculate coefficient of G_m
def calCoef(e):
	return (np.log((1-e)/e))/2

# calculate normlization factor Z
def calNormFactor(alpha, D, regr, dataMat, labelMat):
	Z = 0
	for i in range(len(dataMat)):
		Z += D[i]*np.exp(-alpha*labelMat[i]*regr.predict(dataMat[i].reshape(1, -1)))
	return Z

# updata weights of train samples
def updataWeights(alpha, D, regr, dataMat, labelMat):
	for i in range(len(D)):
		D[i] = D[i]*np.exp(-alpha*labelMat[i]*regr.predict(dataMat[i].reshape(1, -1)))
	return D

# main function of AdaBoost
# input: number of baseLearner
#	     dataMat, labelMat (in crossvalidation, it contains 9/10 of dataset)
# output: each baseLearner G_i and its weight alpha_i
def AdaBoost(num_of_base_learner, dataMat, labelMat):
	N = len(dataMat)           # number of samples
	D = np.ones((N)) / N       # init weights of train samples (1/N)
	G = []                     # list of baseLearner
	alphaList = []             # weight of each baseLearner
	for m in range(num_of_base_learner):
		regr = linear_model.LogisticRegression()              
		regr.fit(dataMat, labelMat, D)                           # train baseLearner, sample_weight is D
		e = errorRate(D, regr, dataMat, labelMat)                # calculate error rate
		alpha = calCoef(e)                                       # calculate coefficient of G_m
		Z = calNormFactor(alpha, D, regr, dataMat, labelMat)     # calculate normlization factor Z
		D = updataWeights(alpha, D, regr, dataMat, labelMat) / Z # updata weights of train samples
		G.append(regr)
		alphaList.append(alpha)
	return G, alphaList

# pre-process (replace all 0 to -1)
def preProcess(labelMat):
	return np.array([1 if a==1 else -1 for a in labelMat])

# get result from final learner
def getResult(num_of_base_learner, folder, dataMat, testSet, G, alpha):
	filename = 'base'+str(num_of_base_learner)+'_fold'+str(folder)+'.csv'
	dirname = 'experiments'
	fileLoc = os.path.join(dirname, filename)
	outFile = open(fileLoc, 'w')
	for idx in range(len(testSet)):
		G_sum = 0
		for i in range(len(G)):
			G_sum += alpha[i]*G[i].predict(dataMat[testSet[idx]].reshape(1, -1))
		outFile.write(str(testSet[idx]+1)+','+('1' if G_sum>0 else '0')+'\n')
	outFile.close()

t0 = time()
numOfBaseLearner = [1, 5, 10, 100]
dataMat, labelMat = readInData('data.csv', 'targets.csv')  
labelMat = preProcess(labelMat)               # need pre-process (replace all 0 to -1)! 
warnings.filterwarnings("ignore")             # ignore warnings

for num_of_base_learner in numOfBaseLearner:
	kf = KFold(n_splits=10)
	folder = 1
	print('Start base '+str(num_of_base_learner)+'...')
	for trainSet, testSet in kf.split(dataMat):
		print('Start folder '+str(folder)+'...')
		G, alpha = AdaBoost(num_of_base_learner, dataMat[trainSet], labelMat[trainSet])
		getResult(num_of_base_learner, folder, dataMat, testSet, G, alpha)          # write result in files
		folder += 1
print("Done in %0.3fs" %(time()-t0))
