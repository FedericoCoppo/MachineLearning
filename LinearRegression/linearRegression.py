#!/usr/bin/env python

# author: Coppo Federico from Edx IBM course "Machine Learning with Python" 
# 10/11/2019 

# readme:
# waiting for 3.2.0 install the following release candidate
# $ pip install matplotlib==3.2.0rc1

import matplotlib.pyplot as plt
import matplotlib as math
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

# calibration to enable/ disable features
enablePlotFeatures = False
enablePlotFeaturesVsLabel = False
enableMultiLinRegression = True

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
train = cdf[msk] # train datataset
test = cdf[~msk] # test dataset

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

if enableMultiLinRegression == True:
    ## MULTIPLE LINEAR REGRESSION MODEL
    print ("\n\n..START MULTI LINEAR REGRESSION..")
    multi_regr = linear_model.LinearRegression()

    # select other features (FUELCONSUMPTION_CITY and FUELCONSUMPTION_HWY)
    cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','CO2EMISSIONS']]

    # mantain previous mask 
    train = cdf[msk] # train datataset
    test = cdf[~msk] # test dataset

    #train the model
    multi_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
    multi_y = np.asanyarray(train[['CO2EMISSIONS']])
    multi_regr.fit (multi_x, multi_y)

    # The coefficients
    print ('Following coeff and intercept are param of the fit line.\nScikit-learn uses "Ordinary Least Squares" method to solve\n')
    print ('Multi linear coefficients: ', multi_regr.coef_)
    print ('Multi linear intercept: ',multi_regr.intercept_)

    # prediction (test the model)
    y_hat= multi_regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
    x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
    y = np.asanyarray(test[['CO2EMISSIONS']])
    print("Residual sum of squares: %.2f" % np.mean((y_hat - y) ** 2))

    # Explained variance(𝑦,𝑦̂ )=1−𝑉𝑎𝑟{𝑦−𝑦̂ }/𝑉𝑎𝑟{𝑦}
    print('Variance score: %.2f' % multi_regr.score(x, y)) #1 is perfect prediction

print ("Program has terminated !")
 

