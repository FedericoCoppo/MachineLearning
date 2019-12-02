# author: Coppo Federico from Edx IBM course "Machine Learning with Python" 
# 19/11/2019 

# readme:
# waiting for 3.2.0 install the following release candidate
# $ pip install matplotlib==3.2.0rc1

# RECAP 

# Logistic Regression
# To estimate the class of a data point, we need indication on the most probable class for that data point.  

# linear regression y=𝜃0+𝜃1𝑥1+𝜃2𝑥2+⋯   ->  ℎ𝜃(𝑥)=𝜃𝑇𝑋

# logistic regression:
# -> regression -> 𝜎(𝜃𝑇𝑋)
# -> logistic function 𝜎(𝜃𝑇𝑋)
# -> logistic regression: 𝑃𝑟𝑜𝑏𝑎𝑏𝑖𝑙𝑖𝑡𝑦𝑂𝑓𝑎𝐶𝑙𝑎𝑠𝑠1=𝑃(𝑌=1|𝑋)=𝜎(𝜃𝑇𝑋)=𝑒𝜃𝑇𝑋/(1+𝑒𝜃𝑇𝑋)

# calibration
enablePrintDataset = False;

# required library 
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# modules for logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_similarity_score

# modules for confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
import itertools

# module for log loss
from sklearn.metrics import log_loss

# dataset contains telecommunications company's customers info and should be used to reduce the turnover.
churn_df = pd.read_csv("ChurnData.csv")

if enablePrintDataset == True:
    print(churn_df)
    
    # select some features for the modeling.
    churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]

    # change the target data type (integer) as required by the skitlearn.
    churn_df['churn'] = churn_df['churn'].astype('int')

    # print dataset information
    print("dataset row #:", churn_df.shape[0])
    print("dataset column #:", churn_df.shape[1])
    print("dataset column name:",  churn_df.columns[0:churn_df.shape[1]])

# define x and y into the dataset
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])

# normalize the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)

# train/test dataset split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# logistic regression modeling (train): solver is algo to use in the optimization problem
# C parameter is inverse of regularization strengt; it must be a positive float (small values means stronger regularization) 
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

# prediction (test, 40 sample)
y_predicted = LR.predict(X_test)
print("y predicted", y_predicted)
print("y      test",y_test)

# probability: P(Y=1|X), 1-P(Y=1|X), 40 sample
y_predicted_prob = LR.predict_proba(X_test)

# MODEL EVALUATION

# JACCARD INDEX for accuracy evaluation: j (y,y^) = |y intersecated y^| / ( |y| +|y^| - (y inters y^) ); 
# near 1.0 means good accuracy

print( "jaccard index:", jaccard_similarity_score(y_test, y_predicted))

# CONFUSION MATRIX

#
# This function prints and plots the confusion matrix.
# Normalization can be applied by setting `normalize=True`.
#
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
confMatrix = confusion_matrix(y_test, y_predicted, labels=[1,0])
print("\nCONFUSION MATRIX:\n")
print("  prediction  ->   1 0")
print(" true label 1 -> ", confMatrix[0])
print(" true label 0 ->", confMatrix[1])

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(confMatrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

# precision = TP/(TP+FP) is is prediction accuracy
# recall = TP/(TP+FN) is true positive rate
print (classification_report(y_test, y_predicted))

# LOG LOSS
print("Log loss:", log_loss(y_test, y_predicted_prob))

################################################################################################################################
# Logistic Regression model has been rebuild again for the same dataset, with different __solver__ and __regularization__ values
################################################################################################################################

# test
LR2 = LogisticRegression(C=0.01, solver='newton-cg').fit(X_train,y_train)

# probability: P(Y=1|X), 1-P(Y=1|X)
y_predicted_prob = LR2.predict_proba(X_test)

# log loss
print("Log loss using newton-cg and C = 0.01:", log_loss(y_test, y_predicted_prob))
