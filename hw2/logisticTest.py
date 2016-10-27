#! /usr/bin/env python
import sys
import pandas as pd
import numpy as np
eta = 0.000000001

def logistic(z):
	return 1/(1+np.exp(-z+eta))
	
model = open(sys.argv[1])

line = model.readline()

w = np.fromstring(line, sep = ',')

line = model.readline()

b = np.fromstring(line, sep = ',')

test_data = pd.read_csv(sys.argv[2],index_col = 0, header = None)
test_data = test_data.ix[1:,:].as_matrix()
test_data = (test_data - test_data.mean(axis = 0)) / test_data.std(axis = 0)
test_data = logistic(test_data.dot(w.T) +b)

out = open(sys.argv[3],'w+')
out.write('id,label\n')
for i in range(1,601):
	out.write(str(i))
	if test_data[i-1] > 0.5:
		out.write(',1\n')
	else:
		out.write(',0\n')
out.close()
