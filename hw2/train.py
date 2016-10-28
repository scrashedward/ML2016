#! /usr/bin/env python
import sys
import pandas as pd
import numpy as np
eta = 0.000000001

def logistic(z):
	return 1/(1+np.exp(-z+eta))

def error(x, w, b, y):
	return (-(y*np.log(logistic(x.dot(w)+b)+eta) + (1-y)*np.log(1-logistic(x.dot(w)+b)+eta))).mean()

def gradDes(x, w, b, y):
	return (-(y-logistic(x.dot(w)+b)).transpose()*x.transpose()).mean(axis = 1)

def gradDesb(x, w, b, y):
	return (-(y-logistic(x.dot(w)+b))).mean() 

#read data from csv
data = pd.read_csv(sys.argv[1], index_col = 0, header = None)

#get training data from data
#train_data is a numpy ndarray of size 4001, 57 
train_data = data.ix[1:, :57].as_matrix()
#train_data = normalize(train_data, axis = 0)
train_data = (train_data - train_data.mean(axis = 0)) / train_data.std(axis = 0)

#y is the result ndarray of size 4001, 1
y = data.ix[1:, 58].as_matrix()

model = open(sys.argv[2], 'w+')

w = np.zeros(57)
b = 0
alpha = 5 #learning rate
iternum = 100000; #iteration number
gdwSum = np.zeros(57)
gdbSum = 0
l = 0.0001

for roundNum in range(iternum):
	#random columns for stochastic gradient descent
	r = np.random.randint(4001,size=200)
	#calculate gradient descent for w and b
	gdw = gradDes(train_data[r,:], w, b, y[r])
	gdb = gradDesb(train_data[r,:], w, b, y[r])
	#adagrad for w and b
	gdwSum = gdwSum + np.square(gdw)
	gdbSum = gdbSum + np.square(gdb)
	#update parameter value
	w = w - alpha * gdw / np.sqrt(gdwSum+eta).astype(float)
	b = b - alpha * gdb / np.sqrt(gdbSum+eta).astype(float)
	if roundNum % 10000 == 0:
		print roundNum
		print error(train_data, w, b, y)
		#raw_input()

np.savetxt(model, w[:, None].T, delimiter=',')
model.write(str(b)+'\n')
'''
test_data = pd.read_csv('spam_test.csv',index_col = 0, header = None)
test_data = test_data.ix[1:,:].as_matrix()
#test_data = normalize(test_data, axis = 0)
test_data = (test_data - test_data.mean(axis = 0)) / test_data.std(axis = 0)
test_data = logistic(test_data.dot(w) +b)
print test_data.shape
#print test_data


out = open('logistic'+str(alpha) + '.' + str(iternum) +'.csv','w+')
out.write('id,label\n')
for i in range(1,601):
	out.write(str(i))
	if test_data[i-1] > 0.5:
		out.write(',1\n')
	else:
		out.write(',0\n')
out.close()
'''