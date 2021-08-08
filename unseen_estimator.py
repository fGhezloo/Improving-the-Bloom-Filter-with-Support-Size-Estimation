import numpy as np
import math
from scipy.stats import poisson
from scipy.optimize import linprog
from scipy.stats import poisson
from collections import Counter
import pandas as pd
def unseen(f):
	# Input: fingerprint f, where f(i) represents number of elements that appear i times in a sample. 
	# Thus sum_i i*f(i) = sample size. File makeFinger.m transforms a sample into the associated 
	# fingerprint.Output: approximation of 'histogram' of true distribution. Specifically, histx(i) 
	# represents the number of domain elements that occur with probability x(i). Thus sum_i x(i)*histx(i) = 1, 
	# as distributions have total probability mass 1. An approximation of the entropy of the 
	# true distribution can be computed as: Entropy = (-1)*sum(histx.*x.*log(x))
	gridFactor = 1.1 # the grid of probabilities will be geometric with this ratio
	alpha = 0.5 # 0.5 worked well in all examples we tried. if alpha between 0.25 and 1, decreasing alpha increases the chances of overfitting
	maxLPIters = 1000 
	
	f = np.transpose(f)
	#total sample size
	
	k=np.dot(f, np.transpose(range(np.size(f))))
	
	# xLPmin = 1.0/(k*max(10, k))

	# min_i = min(np.where(f>0)[0])

	# if min_i > 1:
	# 	xLPmin = 1/25000#1.0*min_i/k
	xLPmin = 1/25000
	x=[0]
	histx=[0]

	fLP = np.zeros((1, np.size(f)))

	for i in range(1, np.size(f)):
		if f[i]>0:
			wind = [max(1, i-math.ceil(np.sqrt(i))), min(i+math.ceil(np.sqrt(i)), np.size(f))]
			
			if np.sum(f[int(wind[0]):int(wind[1])])<np.sqrt(i):
				x.append(i/k)
				histx.append(f[i])
				fLP[0, i]=0
			else:
				fLP[0, i]=f[i]
		
	#if no LP portion, return the empirical histogram
	
	fmax = np.max(np.where(fLP[0]>0))
	if np.size(fmax)==0:
		x=x[1:]
		histx=histx[1:]
	
	# set up the first LP
	# import pdb
	# pdb.set_trace()
	histx = np.asarray(histx).reshape(1, -1)
	x = np.asarray(x).reshape(1, -1)

	LPmass = 1-np.dot(x, np.transpose(histx))
	
	fLP=[fLP[0, :fmax], np.zeros(1, np.ceil(np.sqrt(fmax)))]
	szLPf=len(fLP[0])
	xLPmax = 1.0*fmax/k
	
	xLP=xLPmin*np.power(gridFactor, range(int(math.ceil(np.log(xLPmax/xLPmin)/np.log(gridFactor)))))
	szLPx = np.size(xLP)
	
	objf=np.zeros((szLPx+2*szLPf, 1))
	
	for i in range(int(math.floor((objf.shape[0]-szLPx)/2))):
		objf[szLPx+1:2*i+szLPx, 0]=1/(np.sqrt(fLP[0][i]+1))
		objf[szLPx+2:2*i+szLPx, 0]=1/np.sqrt(fLP[0][i]+1)
	
	A=np.zeros((2*szLPf, szLPx+2*szLPf))

	B=np.zeros((2*szLPf, 1)).reshape(-1, 1)

	
	for i in range(szLPf):
		
		A[2*i, :szLPx] = poisson.pmf(i, (k*xLP).tolist())#np.random.poisson((k*xLP).tolist(), size=(1, len(xLP)))
		A[2*i+1, :szLPx] = (-1)*A[2*i, :szLPx]
		A[2*i, szLPx+2*i]=-1
		A[2*i+1, szLPx+2*i+1]=-1
		B[2*i, 0]=fLP[0][i]
		B[2*i+1, 0]=-fLP[0][i]
		
	
	Aeq = np.zeros((1, szLPx+2*szLPf)).reshape(1, -1)
	Aeq[0, :szLPx]=xLP
	beq = LPmass
	beq = beq.reshape(-1, 1)
	
	for i in range(szLPx):
		A[:,i]=A[:,i]/xLP[i]
		Aeq[0, i]=Aeq[0, i]/xLP[i]
	
	# import pdb
	# pdb.set_trace()
	res = linprog(np.squeeze(objf), A, B, Aeq, beq, bounds=(0, None), options={"MaxIter": 1000, "disp":False})
	
	
	if res.status ==0:
		print("Optimization terminated successfully")
	if res.status ==1:
		print("Iteration limit reached")
	if res.status==2:
		print("Problem appears to be infeasible")
		return res.x, histx, res.status
	if res.status==3:
		print("Problem appears to be unbounded")
	

	# solve the 2nd LP, which minimizes support size subject to incurring at most
	# alpha worse objective function value (of the objective function in the previous LP). 
	# if res.status==0:
		# objf2=0.0*objf
		# objf2[:szLPx, 0]=1
		
		# A2=np.concatenate((A, np.transpose(objf2)), axis=0)
		# fval = np.dot(np.transpose(objf), res.x.reshape(-1, 1))
		# b2 =  np.concatenate((B, (fval[0]+np.array([alpha])).reshape(1, 1)), axis=0)
		# res2 = linprog(np.squeeze(objf2), A2, b2, Aeq, beq, bounds=(0, None), options={"MaxIter": 1000, "disp":False})
		# if res2.status ==0:
		# 	print("LP2 Optimization terminated successfully")
		# if res2.status ==1:
		# 	print("LP2 Iteration limit reached")
		# if res2.status==2:
		# 	print("LP2 Problem appears to be infeasible")
		# if res2.status==3:
		# 	print("LP2 Problem appears to be unbounded")
	
	res2=res
	res2.x[:szLPx]=res2.x[:szLPx]/xLP.T
	x=np.concatenate((np.array(x).reshape(-1, 1), xLP.reshape(-1, 1)), axis=0)
	histx=np.concatenate((histx, np.transpose(res2.x).reshape(1, -1)), axis=1)

	ind=x.argsort(axis=0)
	histx=histx[0, ind.reshape(1, -1)]
	ind = np.where(histx>0)[1]
	x=x[ind]
	histx=histx[0, ind]
	return x, histx, res.status

def makeFinger(v):
	# import pdb
	# pdb.set_trace()
	# h1 = np.histogram(v, bins=np.arange(1, max(v)+2))
	# counts = Counter(v)
	# labels, values = zip(*counts.items())
	h1= pd.Series(v).value_counts()
	# import pdb
	# pdb.set_trace()
	h1 = h1.to_numpy()
	
	f = np.histogram(h1, bins=np.arange(1, max(h1)+2))
	# import pdb
	# pdb.set_trace()
	f=f[0]
	f=f[1:]
	return f




