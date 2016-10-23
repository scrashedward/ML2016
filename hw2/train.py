#! /usr/bin/env python

import pandas as pd
import numpy as np

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

#y is the result ndarray of size 4001, 1
y = data.ix[1:, 58].as_matrix()

print data.ix[1:,:].as_matrix().shape
print train_data.shape

print logistic(np.array([1, 1]))

w = np.zeros(57)
b = 0
alpha = 0.1 #learning rate
iternum = 1000; #iteration number
gdwSum = np.zeros(57)
gdbSum = 0
eta = 0.0001

for roundNum in range(iternum):
	#calculate gradient descent for w
	gdw = gradDes(train_data, w, b, y)
	print 'gdw'
	print gdw
	raw_input()
	gdb = gradDesb(train_data, w, b, y)
	print 'gdb'
	print gdb
	raw_input()
	gdwSum = gdwSum + np.square(gdw)
	gdbSum = gdbSum + np.square(gdb)
	w = w - alpha * gdw / np.sqrt(gdwSum+eta).astype(float)
	print 'w'
	print w
	raw_input()
	b = b - alpha * gdb / np.sqrt(gdbSum+eta).astype(float)
	print 'b'
	print b
	raw_input()
	print error(train_data, w, b, y)
	raw_input()
