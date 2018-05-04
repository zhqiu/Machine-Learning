# -*- coding: utf-8 -*-
""" Qiu Zihao's homework of ML
    Student ID: 141130077
    Logistic Regression
"""

import numpy as np
import csv
from sklearn.model_selection import KFold


# read in data
def readIndata(datafile, labelfile):
	dataMat = np.genfromtxt(datafile, delimiter=',', dtype=np.float)
	labelMat = np.genfromtxt(labelfile, delimiter=',', dtype=np.int)
	return dataMat, labelMat


# sigmod function
def sigmod(x):
    return 1.0/(1+np.exp(-x))


# calculate Hassian Matrix
def computeHassian(data, predict):
    hessian = []
    n = len(data)

    for i in range(n):
        row = []
        for j in range(n):
            row.append(-data[i]*data[j]*(1-predict)*predict)
        hessian.append(row)
    return hessian


# Classifier base on predict
def classifier(X, omega):
    predict = sigmod(sum(X*omega))
    if predict > 0.5:
        return 1
    else:
        return 0
 

# Newton Iteration
def newtonCg(dataMat, labelMat, times=15): 
	m = len(dataMat)    # number of data in train set 
	n = len(dataMat[0]) # number of attribute in a data 
	
	omega = [0.0]*n   

	while(times):
		gradientSum = [0.0]*n
		hessianMatSum = [[0.0]*n]*n
		for i in range(m):
			predict = sigmod(np.dot(dataMat[i], omega))
			error = labelMat[i]-predict
			gradient = list(np.array(dataMat[i])*((1.0*error)/m))
			gradientSum = list(np.array(gradientSum)+np.array(gradient))
			hessian = computeHassian(dataMat[i], predict/m)
			for j in range(n):
				hessianMatSum[j] = list(np.array(hessianMatSum[j])+np.array(hessian[j])) 
		try:
			hessianMatInv = np.linalg.inv(hessianMatSum)
		except:
			continue;
		for k in range(n):
			omega[k] -= np.dot(hessianMatInv[k], gradientSum)
		times -= 1
	return omega

dataMat, labelMat=readIndata('data.csv', 'targets.csv')

kf = KFold(n_splits=10) 
i=1
j=1
for train, test in kf.split(dataMat):
	csv_file_name='fold'+str(i)+'.csv'
	i += 1
	output=open(csv_file_name, 'w') 
	omega = newtonCg(dataMat[train], labelMat[train])  
	for data in dataMat[test]:
		output.write(str(j)+','+str(classifier(data, omega))+'\n')
		j += 1
