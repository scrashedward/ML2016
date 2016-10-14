# coding=big5
import pandas as pd
import numpy as np
import math
import random

DataSize = 4320
SetSize = 18
SetLength = 10
alpha = 0.5
iternum = 18000
theta = 0.3
eta = 0.001
#b = 3.03583890051
b = 1
w = np.zeros(162)

mask = range(27)
mask.remove(1)
df = pd.read_csv('train.csv', encoding='Big5', usecols = mask, index_col = 'date', na_values='NR')


count = 0
a21db = None
testdb = None
tempArray = None
y = None
for i in range(0,DataSize,SetSize):
	for j in range(16):
		for k in range (18):
			secondArray = df.ix[i+k,str(j):str(j+8)].as_matrix()
			secondArray[np.isnan(secondArray.astype(float))] = 0
			secondArray[secondArray<0] = secondArray[secondArray>=0].mean()
			if k == 0:
				tempArray = secondArray
			else:
				tempArray = np.concatenate((tempArray, secondArray))
		if i==0 and j ==0:
			a21db = np.array([tempArray])
		else:
			a21db = np.concatenate((a21db, [tempArray]))

	if (i/SetSize)%20 != 19:
		for j in range(16,24):
			for k in range(18):
				secondArray = df.ix[i+k,str(j):str(23)].as_matrix()
				tempArray2 = df.ix[i+k+SetSize, str(0):str(j-16)].as_matrix()
				secondArray = np.concatenate((secondArray, tempArray2))
				secondArray[np.isnan(secondArray.astype(float))] = 0
				secondArray[secondArray<0] = secondArray[secondArray>=0].mean()
				if k == 0:
					tempArray = secondArray
				else:
					tempArray = np.concatenate((tempArray, secondArray))
			a21db = np.concatenate((a21db, [tempArray]))
	else:
		a21db = a21db[:-1,:]
	if (i/SetSize)%20 == 0:
		if i == 0:
			y = df.ix[i+9,'9':'23']
			y[y<0] = y[y>=0].mean()
		else:
			y2 = df.ix[i+9,'9':]
			y2[y2<0] = y2[y2>=0].mean()
			y = np.concatenate((y, y2))
	else:
		y2 = df.ix[i+9,'0':]
		y2[y2<0] = y2[y2>=0].mean()
		y = np.concatenate((y, y2))

a21db = a21db.transpose()
print a21db.shape
print y.shape

train = a21db

while 1==1:
	g2 = np.zeros(162)
	gb2 = 0
	for i in range(iternum):
		#r is a random number to identify the data chosen in stochastic gradient descent
		#train is the ndarray of training data, train2 is the stochastic chosen data
		r = random.randint(0, 5252)
		train2 = train[:, r:r+400]
		#c is the array of (y -sigma(x_i*w_i) - b)
		c = (y[r:r+400] - w.dot(train2) - b)
		#g is the gradient descent error matrix and gb is the gradient descent error for b
		g = (2 * (c * -1 * train2).mean(axis = 1))- 2 * theta * w
		gb = (-2 * c.mean())
		#g2 and gb2 is for adagrad
		g2 = g2 + g*g
		gb2 = gb2 + gb*gb
		#the feedback is as following
		w = w - alpha * g  / np.sqrt((g2+eta).astype(float)) 
		b = b - alpha * gb / np.sqrt((gb2+eta).astype(float))
		if i % 100 == 0:
			print i
			error = math.sqrt((((y - w.dot(train) - b)**2).sum())/5652)
			print error
	error =  math.sqrt((((y - w.dot(train) - b)**2).sum())/5652)
	print error

	if error < 100:
		out = open('answer_'+str(error)+'.csv', "w+")
		out.write("id,value\n")
		df2 = pd.read_csv('test_X.csv', na_values='NR', header = None)

		j = 0
		temp2 = None
		for i in range(0, DataSize, SetSize):
			for k in range(18):
				temp = df2.ix[i+k,2:12].as_matrix()
				temp[np.isnan(temp.astype(float))] = 0
				temp[temp<0] = temp[temp>=0].mean()
				if k==0:
					temp2 = temp
				else:
					temp2 = np.concatenate((temp2, temp))
			a = (temp2*w).sum() + b
			out.write('id_'+str(j) + ','+str(a)+'\n')
			j = j + 1
		#out.write(str(b))
		#out.write(',')
		#for i in w[:]:
		#	out.write(str(i)+" ")
		#out.close()
	break;
