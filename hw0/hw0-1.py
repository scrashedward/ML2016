import os
import sys

c = sys.argv[1]
f = sys.argv[2]
tmp = []
dat = open(f, "r")
out = open("ans1.txt", "w+")
line = dat.readline()
while line != "" :
	a = line.split(" ")
	#print float(a[int(c)+1])
	tmp.append(float(a[int(c)+1]))
	line = dat.readline()
tmp.sort()
for i in tmp:
	out.write(str(i)+",")
out.seek(-1, os.SEEK_END)
out.truncate()
