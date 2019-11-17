# author: Coppo Federico from Edx IBM course "Machine Learning with Python" 
# 17/11/2019 

# readme:
# waiting for 3.2.0 install the following relese candidate
# $ pip install matplotlib==3.2.0rc1

# this example show non-linear regression usage
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

#calibration 
enablePlotExample = False
sklearnInstalled = False 		# disable if want to execute code without sklearn package

if sklearnInstalled == True:
	from sklearn.metrics import r2_score

# NON LINEAR FUNCTION EXAMPLE 

x = np.arange(-5.0, 5.0, 0.1)

# example of linear funtion 
y = 1.1*(x) + 7
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'g') 
plt.ylabel('Y')
plt.xlabel('X')
plt.show()

# example of non linear funtion: cubic function (y = ax^3 + bx + c)
y = 1*(x**3) + 1*(x**2) + 1*x + 3
y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Y = ax^3 + bx + c')
plt.xlabel('X')
plt.show()

# example of non linear funtion: quadratic function (y =x^2)
y = np.power(x,2)
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'c') 
plt.ylabel('Y = x^2')
plt.xlabel('X')
plt.show()

# example of non linear funtion: exponential function (y =a+ b*c^x)
y= np.exp(x)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'y') 
plt.ylabel('Y = a +b*c^x')
plt.xlabel('X')
plt.show()

# example of non linear funtion: log function (y = log(x))
# indep variable should start > zero
x = np.arange(0.1, 5.0, 0.1) 
y = np.log(x)
#noise should be re-calculated
y_noise = np.random.normal(size=x.size) 
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'x') 
plt.ylabel('Y = log(X)')
plt.xlabel('X')
plt.show()

# example of non linear funtion: sigmodal, logistic (y = a + b/(1 + c^(x-d))
x = np.arange(-5.0, 5.0, 0.1)
y = 1-4/(1+np.power(3, x-2))
y_noise = np.random.normal(size=x.size) 
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'p') 
plt.ylabel('Y = a + b/(1 + c^(X-d)')
plt.xlabel('X')
plt.show()

# NON LINEAR REGRESSION EXAMPLE 

# dataset
df = pd.read_csv("china_gdp.csv")

#show dataset
print ("dataset: China's corresponding annual gross domestic income in US dollars for that year:\n")
print (df)

#plot dataset
plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

# CHOOSE THE MODEL -> logistic function
# Y^ = 1/(1 + e^(𝛽1-x*𝛽2)) where 𝛽1 controls the curve's steepness and 𝛽2 slides the curve on the x-axis.
if enablePlotExample == True:
    #example 1
    y = 1 / (1 + np.exp(-1*(x-5)))
    plt.plot(x,y, 'p') 
    plt.ylabel('Y = 1/(1+ e(-(x-5))))')
    plt.xlabel('X')
    plt.show()

    #example 2
    y = 1 / (1 + np.exp(-3*(x-8)))
    plt.plot(x,y, 'p') 
    plt.ylabel('Y = 1/(1+ e(-3(x-8))))')
    plt.xlabel('X')
    plt.show()

# BUILD THE MODEL

# routine define
def sigmoid_f(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y

# find the parameter that fit the data
beta_1 = 0.10
beta_2 = 1990.0
Y_pred = sigmoid_f(x_data, beta_1 , beta_2)

#plot initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')

# Lets first normalize our x and y (to find the best parameters for our model).
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

# find the best parameter
# curve_fit routine uses non-linear least squares.
# optimal values -> the sum of the squared residuals of y^ - ydata is minimized.
popt, pcov = curve_fit(sigmoid_f, xdata, ydata)
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

#verify the model
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid_f(x, popt[0], popt[1] )
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

#model accuracy
# split data (xdata, ydata are from dataset) into train/test
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# build the model using train set
popt, pcov = curve_fit(sigmoid_f, train_x, train_y)

# predict using test set (x) and model with parameter calculated for train model
y_predicted = sigmoid_f(test_x, *popt)

# model accuracy
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_predicted - test_y)))    #1/n sum (y -y_)
print("Residual sum of squares (MSE): %.4f" % np.mean((y_predicted - test_y) ** 2))#1/n sum (y -y_)^2
if sklearnInstalled == True:
	print("R2-score: %.2f" % r2_score(y_predicted , test_y) )                      # RMSE 1.0 is good 0.0 is bad 