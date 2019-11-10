#!/usr/bin/env python

# author: Coppo Federico from Edx IBM course "Machine Learning with Python" 
# 10/11/2019 

# readme:
# python 3.8.0
# waiting for 3.2.0 install the following relese candidate:  $ pip install matplotlib==3.2.0rc1
# sklearn lib compatible with python 3.8 is ongoing

import matplotlib.pyplot as plt
import matplotlib as math
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

# calib
enablePlotFeatures = False
enablePlotFeaturesVsLabel = False

#read the data
df = pd.read_csv("FuelConsumption.csv")
print (df)

# select some features 
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

if enablePlotFeatures == True:
	# we can plot each of these features
	viz = cdf[['ENGINESIZE','CYLINDERS','CO2EMISSIONS']]
	viz.hist()
	# plot features graph
	plt.show()

if enablePlotFeaturesVsLabel == True:
	# feature FUELCONSUMPTION_COMB Vs label EMISSION
	plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='green')
	plt.xlabel("FUELCONSUMPTION_COMB")
	plt.ylabel("Emission")
	plt.show()

if enablePlotFeaturesVsLabel == True:
	# feature FUELCONSUMPTION_COMB Vs label EMISSION
	plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
	plt.xlabel("Engine size")
	plt.ylabel("Emission")
	plt.show()

if enablePlotFeaturesVsLabel == True:
	plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='red')
	plt.xlabel("cylinders#")
	plt.ylabel("Emission")
	plt.show()

# SPLIT DATASET into TRAIN dataset and TEST dataset (80% of the entire data for training, and the 20% for testing)
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

print ("train dataset\n")
print (train)
print ("test dataset\n")
print (test)

# linear regression 

# train the model
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='black')
plt.xlabel("Engine size (TRAIN)")
plt.ylabel("Emission (TRAIN)")
plt.show()

# use sklearn package to model data
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)

# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

# plot output
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='yellow')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


# accurancy calculation
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y))) #1/n sum (y -y_)
print("mean square error (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))   #1/n sum (y -y_)^2 
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )                       # RMSE 1.0 is good 0.0 is bad

print (" program has terminated !")
 

