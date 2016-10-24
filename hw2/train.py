#! /usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

def logistic(z):
	return 1/(1+np.exp(-z))

def error(x, w, b, y):
	return (-(y*np.log(logistic(x.dot(w)+b)) + (1-y)*np.log(1-logistic(x.dot(w)+b)))).sum()

def gradDes(x, w, b, y):
	return (-(y-logistic(x.dot(w)+b)).transpose()*x.transpose()).mean(axis = 1)

def gradDesb(x, w, b, y):
	return (-(y-logistic(x.dot(w)+b))*b).mean() 

#read data from csv
data = pd.read_csv('spam_train.csv', index_col = 0, header = None)

#get training data from data
#train_data is a numpy ndarray of size 4001, 57 
train_data = data.ix[1:, :57].as_matrix()
train_data = normalize(train_data, axis = 0)

self_test = data.ix[3802:, :57].as_matrix()
self_trst = normalize(self_test, axis = 0)

#y is the result ndarray of size 4001, 1
y = data.ix[1:, 58].as_matrix()
self_y = data.ix[3802:, 58].as_matrix()

print data.ix[1:,:].as_matrix().shape
print train_data.shape



w = np.zeros(57)
b = 0
alpha = 1 #learning rate
iternum = 10000; #iteration number
gdwSum = np.zeros(57)
gdbSum = 0
eta = 0.0001
l = 0.1

for roundNum in range(iternum):
	#calculate gradient descent for w
	gdw = gradDes(train_data, w, b, y)# - l*w
	gdb = gradDesb(train_data, w, b, y)
	gdwSum = gdwSum + np.square(gdw)
	gdbSum = gdbSum + np.square(gdb)
	w = w - alpha * gdw / np.sqrt(gdwSum+eta).astype(float)
	b = b - alpha * gdb / np.sqrt(gdbSum+eta).astype(float)
	if roundNum % 1000 == 0:
		print roundNum
		print error(train_data, w, b, y)
		#raw_input()
'''
self_test = logistic(self_test.dot(w) + b)
self_test[self_test > 0.5] = 1
self_test[self_test <= 0.5] = 0
print self_test


self_err = 0
for i in range(200):
	if( self_test[i] != self_y[i]):
		self_err = self_err+1

print self_err


'''
test_data = pd.read_csv('spam_test.csv',index_col = 0, header = None)
test_data = test_data.ix[1:,:].as_matrix()
test_data = normalize(test_data, axis = 0)
test_data = logistic(test_data.dot(w) +b)
print test_data.shape
#print test_data

out = open('logistic.csv','w+')
out.write('id,label\n')
for i in range(1,601):
	out.write(str(i))
	if test_data[i-1] > 0.5:
		out.write(',1\n')
	else:
		out.write(',0\n')
out.close()
