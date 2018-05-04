# -*- coding: utf-8 -*-
"""	Qiu Zihao's homework of ML
	Student ID: 141130077
	SVM
	Target: use model and testdata to give predicts
"""
import numpy as np

def calEuclidDisSquare(u ,v):
	dis = 0
	for i in range(len(u)):
		dis += np.square(u[i]-v[i])
	return dis

def rbfKernel(gamma, u, v):
	return np.exp(-gamma*calEuclidDisSquare(u, v))
 
def predict(Xt, model):
	gamma = model.get_params()['gamma']
	support_vectors = model.support_vectors_
	result = [] 
	for line in Xt:  
		predict = model.intercept_[0]
		for i, sv in enumerate(support_vectors): 
			predict += model.dual_coef_[0][i]*rbfKernel(gamma, line, sv)  
		result.append(1 if predict>0 else 0)
	return result


