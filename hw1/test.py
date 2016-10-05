# coding=big5
import pandas as pd
import numpy as np
import math
import random

DataSize = 4320
SetSize = 18
SetLength = 10
alpha = 0.6
iternum = 200000
theta = 0.00001
b = 1
eta = 0.0001
w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 
0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
#w = np.array([-0.0322398212742, -0.0626554822073, 0.242918821615, -0.236551919593, -0.0556195931807, 0.509461405446, -0.594239264946, 0.0985475217012, 1.00566957623 ])
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
		secondArray = df.ix[i+9,str(j):str(j+9)].as_matrix()
		if i==0 and j ==0:
			a21db = np.array([secondArray])
		else:
			a21db = np.concatenate((a21db, [secondArray]))
	if ((i/SetSize)+1)%20 != 0:
		for j in range(15,24):
			secondArray = df.ix[i+9,str(j):str(23)].as_matrix()
			tempArray = df.ix[i+9+SetSize, str(0):str(j-15)].as_matrix()
			secondArray = np.concatenate((secondArray, tempArray))
			a21db = np.concatenate((a21db, [secondArray]))


a21db = a21db.transpose()


train = a21db[:-1,:]
temp = train * train
train = np.concatenate((train, temp))
print train.shape
y = a21db[-1:,:]

while 1==1:
	#b = float(random.randint(-5, 5))
	#w = np.array([random.random()*2, random.random()*2, random.random()*2, random.random()*2,
	#random.random()*2,	random.random()*2, random.random()*2, random.random()*2, random.random()*2,
	#random.random()/10,	random.random()/10, random.random()/10, random.random()/10, random.random()/10,
	#random.random()/10,	random.random()/10, random.random()/10, random.random()/10])
	#print b
	#print w
	g2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
	for i in range(iternum+1):
		if i == 50000:
			alpha = alpha/10
		r = random.randint(0, 5252)
		train2 = train[:, r:r+400]
		c = (y[:,r:r+400] - w.dot(train2) - b)
		g = (alpha * 2 * (c * -1 * train2).mean(axis = 1))
		g2 = g2 + g*g
		#print g2+eta
		w = w - alpha * g  / np.sqrt((g2+eta).astype(float))#- 2 * theta * w
		b = b - (alpha * -2 * c.mean())
		if i % 1000 == 0:
			print i
			error = math.sqrt((((y - w.dot(train) - b)**2).sum())/5652)
			print error
			#if i==5000 and error >8:
			#	break;
	error =  math.sqrt((((y - w.dot(train) - b)**2).sum())/5652)
	print error

	if error < 6.5:
		out = open('answer_'+str(error)+'.csv', "w+")
		out.write("id,value\n")
		df2 = pd.read_csv('test_X.csv', na_values='NR', header = None)

		j = 0
		for i in range(0, DataSize, SetSize):
			temp = df2.ix[i+9,'2':].as_matrix()
			temp2 = temp * temp
			temp = np.concatenate((temp, temp2))
			a = (temp*w).sum() + b
			out.write('id_'+str(j) + ','+str(a)+'\n')
			j = j + 1
		out.write(str(b))
		out.write(',')
		for i in w[:]:
			out.write(str(i)+" ")
		out.close()
	break