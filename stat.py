#!/usr/bin/env python

import sys
from math import sqrt
fo_name = sys.argv[-1]


fo = open(fo_name,"r")
average = 0.0
count = 0.0
data = []
for line in fo:
    this_num = float(line.split()[-1])
    data.append(this_num)

accept = data[-1]
start = int(round(0.125*len(data)))
print start
average = 0.0
std_2 = 0.0
count = 0.0

for i in range(start, len(data)-1):
    average = average + data[i]
    count += 1.0

average = average/(count)

for i in range(start, len(data)-1):
    std_2 += (data[i]-average)*(data[i]-average) 
std =sqrt(std_2/(count-1.0))

print str(fo_name) + ":"
print "accept ratio:",this_num
print "average     :",average
print "std         :",std 
print "stand error :",std/sqrt(count)
fo.close()
