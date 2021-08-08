# test unseen estimator:
import numpy as np
from unseen_estimator import unseen, makeFinger
import random
from collections import Counter
import re
from SyntheticData import dataDistribution

f = open("/home/qiuyuc/cse544/cse544_bloomfilter/HamletWords.txt", "r")
data = f.read()
input_list = re.split("\n", data)
n=10000

samp = random.sample(input_list, n)
samp = samp
# zipfDist = dataDistribution(2, [1.5])

# samp = zipfDist.sample(n)
f = makeFinger(samp)
# k=10000
# n=1000
# samp = np.random.randint(n, size=(k, 1))+1

# # compute corresponding 'fingerprint'
# samp = ['abc'+str(x[0]) for x in samp]
# f = makeFinger(samp)
# print(samp[0:5])
x, histx, status = unseen(f)
if status==0:
	# output support size (# species) of the recoverred distribution
	suppSz = np.sum(histx)
	print("find support size: ", suppSz)
else:
	print("couldn't find support size")

