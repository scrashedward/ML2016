#! /usr/bin/env python
import sys
import numpy as np
import pandas as pd
from time import gmtime, strftime

print strftime("%Y-%m-%d %H:%M:%S", gmtime())

eta = 0.00000001

def logistic(z):
	return 1/(1+np.exp(-z+eta))
	
test_data = pd.read_csv(sys.argv[2],index_col = 0, header = None)
test_data = test_data.ix[1:,:].as_matrix()
test_data = (test_data - test_data.mean(axis = 0))/test_data.std(axis = 0)
test_data = np.column_stack((test_data, np.ones(600)))

model = open(sys.argv[1], 'r')

line = model.readline()
test_result = np.zeros(600)
adbNum = 0;
while line != '':
	adbNum = adbNum + 1
	lines = []
	wlines = []
	while line != '# =\n':
		lines.append(line)
		line = model.readline()
	line = model.readline()
	while line != '# =\n':
		wlines.append(line)
		line = model.readline()
	line = model.readline()
	
	w_input = np.zeros((len(lines),58))
	w = np.zeros(len(wlines))
	for i in range(len(lines)):
		w_input[i] = np.fromstring(lines[i], sep = ',')
	for i in range(len(wlines)):
		w[i] = np.fromstring(wlines[i], sep = ',')
	
	test_input = logistic(test_data.dot(w_input.T))
	test_o = logistic(np.column_stack((test_input, np.ones(600))).dot(w))
	test_o[test_o < 0.5] = 0
	test_o[test_o >= 0.5] = 1
	test_result = test_result + test_o

test_o[test_result > float(adbNum)/float(2)] = 1
test_o[test_result < float(adbNum)/float(2)] = 0

#print test_o

out = open(sys.argv[3],'w+')
out.write('id,label\n')
for i in range(1,601):
	out.write(str(i) + "," + str(int(test_o[i-1])) + '\n')
out.close()

print strftime("%Y-%m-%d %H:%M:%S", gmtime())