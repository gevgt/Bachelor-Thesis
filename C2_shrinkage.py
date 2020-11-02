import pandas as pd
import numpy as np 

def shrinkage(x):
	#dimensons
	N = x.shape[1]
	T = x.shape[0]

	#Shrinkage Target

	#sample covariance Matrix
	m = 1/T*np.matmul(np.transpose(x), np.ones((T,1)))
	cvm = 1/T * np.matmul(np.transpose(x-np.transpose(m)), x-np.transpose(m))

	#calculating the average pairwise correlation
	r = 0
	for i in range(0, N-1):
		for j in range(i+1, N):
			r += cvm[i,j]/(cvm[i,i]*cvm[j,j])**0.5
	rquer = 2/((N-1)*N) * r

	#calculating the new covariances
	#all assets have the same correlation now
	F = np.array([])
	for i in range(0, N):
		f = np.array([])
		for j in range(0, N):
			f = np.append(f,rquer*(cvm[i,i]*cvm[j,j])**0.5)
		F = np.append(F, f)
	F = F.reshape(N,N)


	#Shrinkage Intensity

	#pi-hat
	pihat = 0
	pihatm = np.array([])
	for i in range(0,N):
		for j in range(0,N):
			piij = 0
			for t in range(0,T):
				piij += ((x[t,i]-m[i])*(x[t,j]-m[j])-cvm[i,j])**2
			pihatm = np.append(pihatm, piij/T)
			pihat += piij/T
	pihatm = pihatm.reshape(N,N)

	#rho-hat
	part1 = 0
	for i in range(0,N):
		part1 += pihatm[i,i]

	#theta(ii,ij)-hat
	part2 = 0
	for i in range(0,N):
		for j in range(0,N):
			if j == i:
				pass
			else:
				thetaiiij = 0
				thetajjij = 0
				for t in range(0,T):
					thetaiiij += ( (x[t,i]-m[i])**2 - cvm[i,i] ) * ( (x[t,i]-m[i])*(x[t,j]-m[j]) - cvm[i,j] ) /T
					thetajjij += ( (x[t,j]-m[j])**2 - cvm[j,j] ) * ( (x[t,i]-m[i])*(x[t,j]-m[j]) - cvm[i,j] ) /T
				part2 += (rquer/2) * ( (cvm[j,j]/cvm[i,i])**0.5 * thetaiiij + (cvm[i,i]/cvm[j,j])**0.5 * thetajjij )
	rhohat = part1 + part2

	#gammahat
	gammahat = 0
	for i in range(0,N):
		for j in range(0,N):
			gammahat += (F[i,j]-cvm[i,j])**2

	#shrinkage intensity delta-hat
	kappa = (pihat - rhohat) / gammahat
	deltahat = max(0, min(kappa/T,1))


	#Shrunken Matrix
	cvmnew = deltahat * F + (1-deltahat) * cvm
	return cvm, cvmnew
