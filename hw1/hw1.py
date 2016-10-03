# coding=big5
import pandas as pd
import numpy as np
import math
import random

DataSize = 4320
SetSize = 18
SetLength = 10
alpha = 0.000000001
iternum = 30000
theta = 0.00001
b = 3.03583890051
#w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
w = np.array([-0.0322398212742, -0.0626554822073, 0.242918821615, -0.236551919593, -0.0556195931807, 0.509461405446, -0.594239264946, 0.0985475217012, 1.00566957623 ])
#w = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
#w = np.array([0.1, 0.1, 0.1, 0, 0, 0.1, 0, 0.1, 0.7])

mask = range(27)
mask.remove(1)
df = pd.read_csv('train.csv', encoding='Big5', usecols = mask, index_col = 'date', na_values='NR')

'''
First use linear regression to learn pm2.5 from other attributes
concat to a very large array of data
'''
count = 0
a21db = None
testdb = None
for i in range(0,DataSize,SetSize):
	for j in range(15):
#		count = count + 1
#		if count == 5:
#			count = 0
		secondArray = df.ix[i+9,str(j):str(j+9)].as_matrix()
		if i==0 and j ==0:
			a21db = np.array([secondArray])
		else:
			a21db = np.concatenate((a21db, [secondArray]))
	if ((i/SetSize)+1)%20 != 0:
#		count = count + 1
#		if count == 5 :
#			count = 0
		for j in range(15,24):
			secondArray = df.ix[i+9,str(j):str(23)].as_matrix()
			tempArray = df.ix[i+9+SetSize, str(0):str(j-15)].as_matrix()
			secondArray = np.concatenate((secondArray, tempArray))
			a21db = np.concatenate((a21db, [secondArray]))

#	print a21db
#	print i
#where_are_NaNs = np.isnan(a21db)
#a21db[where_are_NaNs] = 0

a21db = a21db.transpose()
#print a21db
#print a21db.shape

train = a21db[:-1,:]
y = a21db[-1:,:]

#print train
#print y.shape
while 1==1:
	#b = float(random.randint(-5, 5))
	#w = np.array([-0.6+random.random()*2, -0.6+random.random()*2, -0.6+random.random()*2, -0.8+random.random()*2, -1.0+random.random()*2, -0.8+random.random()*2, -1.0+random.random()*2, random.random()*2, -0.3+random.random()*2])
	#print b
	#print w
	for i in range(iternum):
		#print 
		c = (y - w.dot(train) - b)
		#c = w.dot(a21db)
		w = w - (alpha * 2 * (c * -1 * train).sum(axis = 1)) - 2 * theta * w
		b = b - (alpha * 2 * c.sum())
		#raw_input()
		if i % 100 == 0:
			print i
			error = math.sqrt((((y - w.dot(train) - b)**2).sum())/5650)
			print error
			if i==1000 and error >8:
				break;
	error =  math.sqrt((((y - w.dot(train) - b)**2).sum())/5650)
	print error

	if error < 6.5:
		out = open('answer_'+str(error)+'.csv', "w+")
		out.write("id,value\n")
		df2 = pd.read_csv('test_X.csv', na_values='NR', header = None)

		j = 0
		for i in range(0, DataSize, SetSize):
			temp = df2.ix[i+9,'2':].as_matrix()
			a = (temp*w).sum() + b
			out.write('id_'+str(j) + ','+str(a)+'\n')
			j = j + 1
		out.write(str(b))
		out.write(',')
		for i in w[:]:
			out.write(str(i)+" ")
		out.close()
		break;