import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import D1_data, D2_monthly, C1_blacklitterman, C2_shrinkage, D3_dates

#data
mcstocks, stockreturns, bondreturns, rf = D1_data.daten()		#daily returns + market cap. weights	
mreturn = D2_monthly.monthly()						#monthly returns 
dates = D3_dates.dates()						#dates from 07.31.2014 to 04.29.2019

#adjusting data set for stockreturns since the first 207 days are used for estimating required Black Litterman inputs
stockreturns = stockreturns[207:]
bondreturns = bondreturns[207:] 
y = np.concatenate((stockreturns, bondreturns), axis = 1)

#dates to which asset allocation is updated
period = np.array([0,	65,	126,	188,	252,	316,
	377,	440,	503,	568,	630,	691,	755,	820,
	882,	943,	1007,	1072,	1133])		

#variables						
j = 0						#every three month, weights are updated; j determines the 10 monthly returns required
						#to calculate the historic average as view for the Black Litterman model
cpfv = 1000					#Portfolio Size in $; required to scale chart
threshold = 0.03				#threshold when rebalancing is triggered
pfv = np.array([cpfv])				#Array of Portfolio Values for each point in time in $
sumdev = 0					#sum of deviation to later calculate transaction costs
count = 0

#this loop calculates the current portfolio values for each day between 08.01.14 - 04.29.19
for i in range(y.shape[0]):
	if i in period:						#if i in period, three month are over and the asset weights need to be updated
		x = mreturn[j:j+10]
		cvm, sigma = C2_shrinkage.shrinkage(mreturn)				#shrinks covariance matrix
		wmarket, neww = C1_blacklitterman.bl(mcstocks[147+i:i+207], x, sigma)	#calculation of optimal asset weights		
		wopt = np.concatenate((neww*0.6, [0.08, 0.08, 0.08, 0.08, 0.08]))	#Strategic asset weights: Equity(60%) + Bonds(40%)
		av = wopt * cpfv				#after update, weights are reset to new asset allocation
		if j != int(period[0]):
			sumdev += np.sum(abs(cw-wopt))
			count += 1
		j += 3						#the next update takes place three months later

	#Rebalancing
	av = av * (1 + y[i,:])				#current asset values
	cpfv = np.sum(av)				#current portfolio value
	pfv = np.append(pfv, cpfv)			#appending current asset value to array
	cw = av / np.sum(av)				#current asset weights
	dev = cw - wopt 				#current deviation from strategic allocation
	pdrift = np.sum(abs(dev))/2			#portfolio drift
	if float(pdrift) >= threshold:			#if portfolio drifts more than threshold allows, rebalancing is triggered
		av = wopt * cpfv
		sumdev += np.sum(abs(cw-wopt))
		count += 1

#characteristics of times series
pfr = pfv[1:] / pfv[0:-1] -1			#portfolio returns
pfm = np.mean(pfr) * 250			#mean of portfolio returns
pfstd = np.std(pfr) * np.sqrt(250)		#volatility of portfolio returns
sr = pfm / pfstd				#sharpe ratio

#sharpe ratio after costs
tacosts = sumdev * 0.00021/(y.shape[0]/250)		#transaction costs: sum of deviations * average bid-ask spread scaled for 1 year
eratio = np.array([0.0003, 0.0004, 0.0007, 0.0007, 0.0005, 0.0012, 0.0006, 0.0015, 0.0025, 0.0009, 0.0039,]) #expense ratios
ercost = np.sum(0.6 * eratio[:6] / 6) + np.sum(0.4 * eratio[6:] / 5) #average ETF cost per year
betterment = 0.0025					#betterment managing fees
totc = tacosts + ercost	+ betterment			#total costs
srafterc = (pfm - totc)/pfstd				#sharpe ratio after costs

#plotting line chart
plt.plot(dates, pfv, linewidth = 1)
#plt.show()

#output
print("-- Betterment PF Strategy --")
print("Threshold: " +  str(threshold))
print("Sharpe Ratio: " + str(round(sr,4)))
print("Annualized Mean Return: " + str(round(pfm, 4)))
print("Annualized volatility: " + str(round(pfstd, 4)))
print("Rebalancing Count: " + str(count))
print()
print("AFTER COSTS")
print("Costs: " + str(round(totc,4)))
print("Sharpe Ratio: " + str(round(srafterc,4)))
print("Annualized Mean Return: " + str(round(pfm - totc, 4)))
print()
