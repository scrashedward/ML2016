#! /usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from random import random

eta = 0.00000001

def logistic(z):
	return 1/(1+np.exp(-z+eta))

def error(o,y):
	return (-(y*np.log(o+eta) + (1-y)*np.log(1-o+eta))).mean()

#read data from csv
data = pd.read_csv('spam_train.csv', index_col = 0, header = None)

#get training data from data
#train_data is a numpy ndarray of size 4001, 57 
temp = np.zeros(3801) + 1
train_data = data.ix[1:3801, :57].as_matrix()
train_data = normalize(train_data, axis = 0)
train_data = np.column_stack((train_data, temp.T))


self_test = data.ix[3802:, :57].as_matrix()
self_test = normalize(self_test, axis = 0)

#y is the result ndarray of size 4001, 1
y = data.ix[1:3801, 58].as_matrix()
self_y = data.ix[3802:, 58].as_matrix()

print data.ix[1:,:].as_matrix().shape
print train_data.shape

#Neuron number
nNum = 5
dataNum = 1000
dataLen = 3801

w_input = np.random.rand(nNum,58)*2-1
w = np.random.rand(nNum + 1)*2-1
alpha = 0.5 #learning rate
iterNum = 15000; #iteration number
gdwiSum = np.zeros((nNum,58))
gdwSum = np.zeros(nNum + 1)

for roundNum in range(iterNum):
	#calculate gradient descent for w
	r = np.random.randint(dataLen, size = dataNum)
	o_input = logistic(train_data[r,:].dot(w_input.T))
	o = logistic(np.column_stack((o_input, np.ones(dataNum))).dot(w))
	delta_o = (o-y[r])

	gdw = (np.column_stack((o_input, np.ones(dataNum))).T * delta_o.T).mean(axis = 1)
	gdwSum = gdwSum + np.square(gdw)
	
	
	gdwi = ((train_data[r,:].T).dot(delta_o[:,None].dot(w[:nNum,None].T) * o_input * ( 1 - o_input)))/dataNum
	gdwi = gdwi.T
	gdwiSum = gdwiSum + np.square(gdwi)

	w = w - alpha * gdw / np.sqrt(gdwSum +eta)
	w_input = w_input - alpha * gdwi / np.sqrt(gdwiSum + eta)

	if roundNum % 1000 == 0:
		print roundNum
		o_input = logistic(train_data.dot(w_input.T))
		o = logistic(np.column_stack((o_input, np.ones(dataLen))).dot(w))
		print error(o, y)
		#raw_input()
#print o


self_test = np.column_stack((self_test, np.ones(200)))
self_input = logistic(self_test.dot(w_input.T))
self_o = logistic(np.column_stack((self_input, np.ones(200))).dot(w))

print 'self test error:' + str(error(self_o,self_y))

test_data = pd.read_csv('spam_test.csv',index_col = 0, header = None)
test_data = test_data.ix[1:,:].as_matrix()
test_data = normalize(test_data, axis = 0)
test_data = np.column_stack((test_data, np.ones(600)))

o_input = logistic(test_data.dot(w_input.T))

test_o = logistic(np.column_stack((o_input, np.ones(600))).dot(w))

#print test_o

out = open('nn.csv','w+')
out.write('id,label\n')
for i in range(1,601):
	out.write(str(i))
	if test_o[i-1] > 0.5:
		out.write(',1\n')
	else:
		out.write(',0\n')
out.close()
