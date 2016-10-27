#! /usr/bin/env python

import pandas as pd
import numpy as np
from time import gmtime, strftime
from random import random, randint

print strftime("%Y-%m-%d %H:%M:%S", gmtime())

print 'Program finish after 25 models trained'

eta = 0.00000001

def logistic(z):
	return 1/(1+np.exp(-z+eta))

def error(o,y):
	return (-(y*np.log(o+eta) + (1-y)*np.log(1-o+eta))).mean()


#read data from csv
data = pd.read_csv('spam_train.csv', index_col = 0, header = None)

#get training data from data
#train_data is a numpy ndarray of size 4001, 57 
temp = np.zeros(4001) + 1
train_data = data.ix[1:, :57].as_matrix()
#train_data = normalize(train_data, axis = 0)
#train_data = train_data / (np.linalg.norm(train_data, ord = 2, axis = 0))
train_data = (train_data - train_data.mean(axis = 0))/train_data.std(axis = 0)
train_data = np.column_stack((train_data, temp.T))

#y is the result ndarray of size 4001, 1
y = data.ix[1:, 58].as_matrix()

model = open('model', 'w+')
adbNum = 25
w_input = []
w = []
test_result = np.zeros(600)
for i in range(adbNum):

	#Neuron number
	nNum = np.random.randint(low = 2, high = 25)
	dataNum = 1000
	dataLen = 4001

	w_input.append(np.random.rand(nNum,58)*2-1)
	w.append(np.random.rand(nNum + 1)*2-1)
	alpha = 1.5  #learning rate
	iterNum = np.random.randint(low = 12, high = 18) * 1000; #iteration number
	gdwiSum = np.zeros((nNum,58))
	gdwSum = np.zeros(nNum + 1)

	for roundNum in range(iterNum):
		#calculate gradient descent for w
		r = np.random.randint(dataLen, size = dataNum)
		o_input = logistic(train_data[r,:].dot(w_input[i].T))
		o = logistic(np.column_stack((o_input, np.ones(dataNum))).dot(w[i]))
		delta_o = (o-y[r])

		gdw = (np.column_stack((o_input, np.ones(dataNum))).T * delta_o.T).mean(axis = 1)
		gdwSum = gdwSum + np.square(gdw)
		
		
		gdwi = ((train_data[r,:].T).dot(delta_o[:,None].dot(w[i][:nNum,None].T) * o_input * ( 1 - o_input)))/dataNum
		gdwi = gdwi.T
		gdwiSum = gdwiSum + np.square(gdwi)

		w[i] = w[i] - alpha * gdw / np.sqrt(gdwSum +eta)
		w_input[i] = w_input[i] - alpha * gdwi / np.sqrt(gdwiSum + eta)

		if roundNum == iterNum - 1:
		#	print roundNum
			o_input = logistic(train_data.dot(w_input[i].T))
			o = logistic(np.column_stack((o_input, np.ones(dataLen))).dot(w[i]))
			print str(i) + ": " + str(error(o, y))
		#	raw_input()
	np.savetxt(model, w_input[i], delimiter = ',', footer = '=')
	np.savetxt(model, w[i], delimiter = ',', footer = '=')

print strftime("%Y-%m-%d %H:%M:%S", gmtime())