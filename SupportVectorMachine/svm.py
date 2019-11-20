# author: Coppo Federico from Edx IBM course "Machine Learning with Python" 
# 20/11/2019 

# readme:
# waiting for 3.2.0 install the following relese candidate
# $ pip install matplotlib==3.2.0rc1

# Support Vector Machine

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score

# calibration 
enablePrintDataset = False;

# the dataset consists of several hundred human cell sample records
cell_df = pd.read_csv("cell_samples.csv")

if enablePrintDataset == True:
    print(cell_df)
    
# The Class field is the target that contains the diagnosis (confirmed by a separated procedure)  

# Target class under evaluation: benign (value = 2) or malignant (value = 4)
# below is plotted the distribution of this classes based on Clump(x) and UnifSize(y) features
ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()

# data preprocessing
print("Features data type:\n",cell_df.dtypes)

# change non-numerical data
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
print("Features data type after data selection:\n",cell_df.dtypes)

feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)

# the model should predict only 2 possible value (benign (=2) or malignant (=4))
cell_df['Class'] = cell_df['Class'].astype('int') # copy the array casting to a int
y = np.asarray(cell_df['Class'])                  # convert the input to an array

# split the dataset into train/test set
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# SVM with Scikit-learn

# the algo offers a choice of kernel functions (needed for separation) 
# mapping data into a higher dimensional space (kernelling) is called is needed for separation
# kernel function could be of different types (linear, polynomial, radial basis function (RBF), sigmoid)
# there's no easy way of knowing which function performs best with any given dataset: choose different functions in turn and compare the results is a method

svc_kernel = svm.SVC(kernel='rbf', gamma='scale')

# fit the model with kernel function
svc_kernel.fit(X_train, y_train)

# predict new values
y_predicted = svc_kernel.predict(X_test)

# MODEL EVALUATION

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

# compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_predicted, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, y_predicted))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')

# F1 score = 2*(precision*recall)/(precision+recall)
print("f1 score:", f1_score(y_test, y_predicted, average='weighted'))

# jaccard index
print("jaccard idx:", jaccard_similarity_score(y_test, y_predicted))

##############################################################
# rebuild the model, but this time with a __sigmoid__ kernel  #
##############################################################
svc_kernel_linear = svm.SVC(kernel='sigmoid', gamma='scale')

# fit the model with kernel function
svc_kernel_linear.fit(X_train, y_train)

# predict new values
y_predicted = svc_kernel_linear.predict(X_test)

# new accuracy

# F1 score
print("f1 score (sigmoid):", f1_score(y_test, y_predicted, average='weighted'))

# jaccard index
print("jaccard idx (sigmoid):", jaccard_similarity_score(y_test, y_predicted))
