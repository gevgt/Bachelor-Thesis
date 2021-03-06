import pandas as pd 
import numpy as np

def bl(mcstocks, x, sigma):
	exp = np.mean(x, axis = 0)						#as views, the average of historic returns is used

	#Equilibrium Returns: pi = lamb*sigma*wmarket
	mcmean = np.mean(mcstocks, axis = 0)					#calculates the average market capitalization weight for each asset
	wmarket = mcmean/np.sum(mcmean)
	pfmean = np.matmul(wmarket, np.transpose(exp))				#portfolio mean -> required for risk aversion coefficient
	pfvar = np.matmul(wmarket, np.matmul(sigma, np.transpose(wmarket)))	#portfolio risk
	lamb = pfmean/pfvar							#risk aversion coefficient to scale implied equilibrium returns

	pi = lamb * np.matmul(sigma,wmarket)					#implied equilibrium returns


	#Q - Vektor of Views, in our Case the Fama-French Returns
	N = x.shape[1]								#N is the number of stocks
	diff = exp - pi 							#difference between views and implied equilibrium returns
	Q = pi + 0.1 * diff 							#10% of difference is added to pi to gently tilt portfolio, as 
										#views are usually to "extreme" -> difference to Black Litterman


	#P - Matrix of Assets involved in a View
	K = N 									#K is the number of views
	P = np.identity(K)							#each view is an absolute view

	#Omega - K x K Matrix of the View´s Uncertainity
	Omega = np.array([])							#implement array to append omegas later on
	teta = 1								#scales the variance of error terms

	for i in range(0, K):							#the loop appends the variance of error terms to the array
		for j in range(0, K):
			if i == j:
				Omega = np.append(Omega, np.matmul(P[i,:], np.matmul(sigma, np.transpose(P[i,:]))) * teta)
			else:
				Omega = np.append(Omega, 0)
	Omega = Omega.reshape(K,K)						#initially Omega is a column vector, but it needs to be a K x K Matrix


	#Black-Litterman Formula: 𝐸[𝑅] = [(𝜏Σ)**-1 + 𝑃' Ω**-1 𝑃]**-1 [(𝜏Σ)**-1 Π + 𝑃' Ω**-1 𝑄]
	part1 = np.linalg.inv( np.linalg.inv(teta*sigma) + np.matmul(np.transpose(P), np.matmul(np.linalg.inv(Omega),P)) )
	part2 = (np.matmul(np.linalg.inv(teta*sigma), pi) + np.matmul(np.transpose(P), np.matmul(np.linalg.inv(Omega), Q)))
	newexp = np.matmul(part1, part2)


	#New optimal Weights
	neww = np.around(np.matmul(np.linalg.inv(lamb*sigma), newexp),4)	#w = (λΣ)**(-1) Π
	neww = neww/np.sum(neww)						#ensure that weights sum up to 1 since absolute views are used

	return wmarket, neww
