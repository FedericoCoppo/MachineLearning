# MachineLearning
tutorial code modified for experimental purposes from edx machine learning  IMB course.

# linear regression  [$ python linearRegression.py]
The .cvs file contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada.
The script use supervised machine learning (regression); it is used the simple linear regression model representation to estimate C02 emission from engine size; it is used the train/test dataset split method.
Then same proccess is done using multi linear regression for FUELCONSUMPTION_CITY and FUELCONSUMPTION_HWY features.

# non linear regression [$ python nonLinearRegression.py]
The .cvs file contains two columns: the first, a year between 1960 and 2014, the second, China's corresponding annual gross domestic income in US dollars for that year.
The script exposes different non linear examples and ((for .cvs datas) uses non linear regression for model creation (with logistic function) and train-test split for model accurancy evalutation. 

# classification K-Nearest Neighbors [$ python classificationKNN.py]
The .cvs file contains telecommunication provider customer dataset (1000 sample) with four target customer-type info.
The script exposes classification model train-test split using Knn algo with different k value.

# decision trees [$ python decisionTrees.py]
The dataset contains set of patients that suffered from the same illness; each patient responded to one of 5 medications target (Drug A,B,C,X,Y). The script try to create a model to find out which drug might be appropriate for a future patient with the same illness using decision trees.

# logistic regression [$ python logisticRegression.py]
The dataset contains telecommunications company's customers info and turnover.
The script uses logistic regression for customer classification model and uses jaccard index, confusion matrix and log loss for model accuracy calculation. 

# support vector machine [$ python svm.py]
The dataset contains human cell data. The script uses support vector machine for benign/malignant cancer classification model (train/test) and uses jaccard index, confusion matrix and f1-score for model accuracy calculation. 

# clustering k-means algo [$ python clusteringKmeans.py]
The dataset contains customer data. The script uses clustering k-means algo (sklearn.cluster) for customer segmentation using both random dataset and customer dataset.

# hierarchical clustering [$ python clusteringHierarchical.py]
The dataset contains vehicle characteristic data. The script uses clustering agglomerative algo for bottom up clustering using both a random genrated dataset and vehicle dataset.

# dbscan clustering [$ python dbscanClustering.py]
The dataset contains weather station data. The script uses "density-based spatial clustering of applications with noise" algo for dataset clustering.

# content base recommender  [$ python contentBaseRecommender.py]
The dataset contains movies info dataset. The script uses content-based recommendation systems algo to recommend movies to users based on movie info taken from the user. in addition the script contains collaborative filtering part example (using the same dataset).
