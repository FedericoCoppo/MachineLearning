# author: Coppo Federico from Edx IBM course "Machine Learning with Python" 
# 18/11/2019 

# readme:
# waiting for 3.2.0 install the following relese candidate
# $ pip install matplotlib==3.2.0rc1

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier    # classification algorithm
from sklearn import metrics

# calibration 
enablePlotDataset = False
enablePlotHisto = False

# classification problem: dataset have customer data of telecom. provider (target is 4 group of target)
df = pd.read_csv('teleCust1000t.csv')

if enablePlotDataset == True:
    print ("dataset: telecommunication provider customers classsification:\n")
    print (df)
    
# plot customer classes number    
print("The dataset contains following classes [1->Basic-service 2-> E-Service customers 3->Plus Service 4->Total Service]:")
df['custcat'].value_counts()

if enablePlotHisto == True:
    # plot histogram of some features ( x assis value of features, y axis the occurences)
    df.hist(column='custcat')
    df.hist(column='age', bins=50)

# convert the pandas DATAFRAME into a numpy ARRAY (is needed to use scikit-learn library)
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
print("first 5 datapoint of all features")
print(X[0:5])

print("first 5 datapoint of target labels")
y = df['custcat'].values
print(y[0:5])

# Data Normalization [X = (xi - mean) / standardDev]
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

# 1000 datapoint in cvs, 800 for train 200 for test: 11 is the number of features
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# Classification of K nearest neighbor (KNN)

k = 4

# train Model
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

# Predict
y_predicted = neigh.predict(X_test)
y_predicted[0:200]

# model accuracy evaluation

# Following function calculates how closely the actual labels and predicted labels are matched in the test set
# it is similar to jaccard index: j (y,y^) = |y intersecated y^| / ( |y| +|y^| - (y inters y^) )
print("Train set Accuracy (k = 4): ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy (k = 4): ", metrics.accuracy_score(y_test, y_predicted))

# repeat traning with K = 6
neigh_k6 = KNeighborsClassifier(n_neighbors = 6).fit(X_train,y_train)

# predict
y_predicted_k6 = neigh_k6.predict(X_test)
y_predicted_k6[0:200]

# evaluate new model
print("Train set Accuracy (k = 6): ", metrics.accuracy_score(y_train, neigh_k6.predict(X_train)))
print("Test set Accuracy (k = 6): ", metrics.accuracy_score(y_test, y_predicted_k6))

# CHOOSE THE CORRECT K fo KNN model
# test ten model (k from 1 to 10) 
Kmax = 10

# array for model accurancy evaluation
mean_acc = np.zeros((Kmax-1))
std_acc = np.zeros((Kmax-1))

# confusione matrix
ConfustionMx = [];

# iterate
for n in range(1,Kmax):    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    y_predicted=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, y_predicted)
    std_acc[n-1]=np.std(y_predicted==y_test)/np.sqrt(y_predicted.shape[0])

print("Mean array (k 1 to 10): ", mean_acc)

# plot model accuracy for Different number of Neighbors
# plot x = kmax, y = accuracy
plt.plot(range(1,Kmax),mean_acc,'g')

# fill the area between two horizontal curves ( accuracy +- standard deviation @ 3sigma)
plt.fill_between(range(1,Kmax),mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)

# update graph legend
plt.legend(('Accuracy ', '+/- 3xstd'))

# update graph tag
plt.ylabel('Accuracy ')
plt.xlabel('K')

# fits subplot in to the figure area
plt.tight_layout()
plt.show
