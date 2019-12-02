# author: Coppo Federico from Edx IBM course "Machine Learning with Python" 
# 18/11/2019 

# readme:
# waiting for 3.2.0 install the following release candidate
# $ pip install matplotlib==3.2.0rc1

# the script try to create a model to find out which drug might be appropriate for a future patient with the same illness

# Notice: You might need to uncomment and install the pydotplus and graphviz libraries if you have not installed these before
# !conda install -c conda-forge pydotplus -y
# !conda install -c conda-forge python-graphviz -y

import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.spatial import distance

# calibration
enablePrintDataset = False

# the dataset contains set of patients (feature sets are age, sex, blood pressure, cholesterol and Na_to_K) that suffered from the same illness
# each patient responded to one of 5 medications target (Drug A,B,C,X,Y)

my_data = pd.read_csv("drug200.csv", delimiter=",")
print("size of patients:", my_data.shape[0])

if enablePrintDataset == True:
    print(my_data)

# X is feature matrix
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

# sklearn decision trees do not handle categorical variables; it should be converted into into dummy variables

# 2nd feature (sex) will become 0 or 1
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 

# 3rd feature (blood pressure) will become 0 or 1 or 2
le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

# 4rt feature (cholesterol) will become 0 or 1
le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

# y is the target variable 
y = my_data["Drug"]

# train/test split
# X and y are the arrays required before the split; the test_size is testing dataset ratio; the random_state ensures that we obtain the same splits
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

# 200 datapoint in cvs, 140 for train, 60 for test: 5 is the number of features (age, sex, blood pressure, cholesterol and Na_to_K)
print ('Train set:', X_trainset.shape,  y_trainset.shape)
print ('Test set:', X_testset.shape,  y_testset.shape)

# model
# the classifier is pecified with "entropy" to see gain info of each node
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
# print(drugTree) # to shows the default parameters

# train the model
drugTree.fit(X_trainset,y_trainset)

# prediction on testing dataset
predTree = drugTree.predict(X_testset)

# prediction Vs actual values.
print(predTree)
print (y_testset)

# model evaluation
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

# model accuracy without sklearn
print( "accuracy evaluation using [scipy.spatial.distance] module instead of [sklearn]:", 1 - distance.jaccard(predTree, y_testset))

# TREE DISPLAY

# Add modules needed for display the tree 
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

# implements a file-like class that reads and writes a string buffer
dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5] # keep the features as list
targetNames = my_data["Drug"].unique().tolist() # keep the target as list

# export a decision tree in DOT format: export_graphviz -> this function generates a GraphViz representation of the decision tree, which is then written into out_file (dot_data)

# create DOT data
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  

# draw graph
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

# write the image to png file (drugtree.png)
graph.write_png(filename)
