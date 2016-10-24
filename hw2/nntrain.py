#! /usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from random import random

def logistic(z):
	return 1/(1+np.exp(-z))

def error(x, w, b, y):
	return (-(y*np.log(logistic(x.dot(w)+b)) + (1-y)*np.log(1-logistic(x.dot(w)+b)))).sum()

def nn(x, w1, w2, w3, w, b1, b2, b3, y):
	o1 = logistic(x.dot(w1)+b1)
	o2 = logistic(x.dot(w2)+b2)
	o3 = logistic(x.dot(w3)+b3)
	return logistic(np.array([a1,a2,a3]).dot(w)+b)
	

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



w1 = np.random.rand(57)
w2 = np.random.rand(57)
w3 = np.random.rand(57)
w = np.random.rand(3)
b = random()
b1 = random()
b2 = random()
b3 = random()
alpha = 1 #learning rate
iternum = 10000; #iteration number
gdw1Sum = np.zeros(57)
gdw2Sum = np.zeros(57)
gdw3Sum = np.zeros(57)
gdwSum = np.zeros(3)
gdbSum = 0
gdb1Sum = 0
gdb2Sum = 0
gdb3Sum = 0
eta = 0.000001
l = 0.1

for roundNum in range(iternum):
	#calculate gradient descent for w
	o1 = logistic(train_data.dot(w1)+b1)
	o2 = logistic(train_data.dot(w2)+b2)
	o3 = logistic(train_data.dot(w3)+b3)
	o = logistic(np.array([o1,o2,o3]).T.dot(w) + b)
#	print np.array([o1,o2,o3]).shape
#	raw_input()
	delta_o = ((o-y)*o*(1-o)).mean()

	gdw = np.array([o1.mean(), o2.mean(), o3.mean()]) * delta_o
	gdwSum = gdwSum + np.square(gdw)
	w = w - alpha * gdw / np.sqrt(gdwSum +eta)
	gdw1 = train_data.mean(axis = 0) * delta_o * o1.mean() * (1-o1.mean())
	gdw1Sum = gdw1Sum + np.square(gdw1)
	w1 = w1 - alpha * gdw1 / np.sqrt(gdw1Sum + eta)
	gdw2 = train_data.mean(axis = 0) * delta_o * o2.mean() * (1-o2.mean())
	gdw2Sum = gdw2Sum + np.square(gdw2)
	w2 = w2 - alpha * gdw2 / np.sqrt(gdw2Sum + eta)
	gdw3 = train_data.mean(axis = 0) * delta_o * o3.mean() * (1-o3.mean())
	gdw3Sum = gdw3Sum + np.square(gdw3)
	w3 = w3 - alpha * gdw3 / np.sqrt(gdw3Sum + eta)

	if roundNum % 1000 == 0:
		print roundNum
#		print error(train_data, w, b, y)
#		#raw_input()
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
test_o1 = logistic(test_data.dot(w1)+b1)
test_o2 = logistic(test_data.dot(w2)+b2)
test_o3 = logistic(test_data.dot(w3)+b3)
test_o = logistic(np.array([test_o1, test_o2, test_o3]).T.dot(w) + b )
print test_o.shape
#print test_data

out = open('nn.csv','w+')
out.write('id,label\n')
for i in range(1,601):
	out.write(str(i))
	if test_o[i-1] > 0.5:
		out.write(',1\n')
	else:
		out.write(',0\n')
out.close()
