# author: Coppo Federico from Edx IBM course "Machine Learning with Python" 
# 26/11/2019 

# readme:
# waiting for 3.2.0 install the following relese candidate
# $ pip install matplotlib==3.2.0rc1

# CLUSTERING using k-Means for customer segmentation

import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D 

# SET UP RANDOM DATA
print("DATASET WITH RANDOM DATA!")

# set up a random seed
np.random.seed(0)

# set up random clusters
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

# display the scatter plot
plt.scatter(X[:, 0], X[:, 1], marker='.')

# K-MEANS algo parameter description:
# centroid init: k-means++
# n_cluster = 4
# number of time the k-means algorithm runs with different centroid seeds: n_init = 12
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)

# fit the k-means model with the feature matrix
k_means.fit(X)

# keep the labels for each point and the coordinates of the cluster centers
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_

# VISUAL PLOT

# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# set the colormap to "spectral"
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# add subplot to the current figure (1,1,1 is the position of the subplot)
ax = fig.add_subplot(1, 1, 1)

# plots the data points and centroids.
# k will range from 0-3 (which will match the possible clusters that each data point is in)
# zip: the first item in each passed iterator is paired together

for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):    
    
    # Create a list of all data points, the data in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=7)

# Title of the plot
ax.set_title('k means algo on random data')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()


# SEGMENT DATASET INTO 2 CLUSTER
k_means3 = KMeans(init = "k-means++", n_clusters = 2, n_init = 12)
k_means3.fit(X)
fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means3.labels_))))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len(k_means3.cluster_centers_)), colors):
    my_members = (k_means3.labels_ == k)
    cluster_center = k_means3.cluster_centers_[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
ax.set_xticks(())
ax.set_yticks(())
plt.show()

# CUSTOMER SEGMENTATION with K-MEANS load data from csv
print("DATASET (Cust_Segmentation.csv) WITH CUSTOMER DATA!")

# calibration
enablePlotDataset = True

# load dataset
cust_df = pd.read_csv("Cust_Segmentation.csv")

if enablePlotDataset == True:
    print ("customer dataset:\n")
    print(cust_df)

# remove address features: address is a categorical variable ans k-means algorithm isn't directly applicable (euclidean distance)
df = cust_df.drop('Address', axis=1)

if enablePlotDataset == True:
    print ("customer dataset after categorical features drop:\n")
    print(df)   
    
# normalizing over the standard deviation to helps algorithms interpreting features with different magnitudes and distributions equally
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)

# apply k-means model on our dataset
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels) # cluster = 3 -> [1 2 1 1....0]; cluster = 4 [0 3 1 0 2 ...]

# assign the labels to each row in the dataframe 
df["Cluster_label"] = labels

# plot new dataset
if enablePlotDataset == True:
    print ("customer dataset after cluster label has been added:\n")
    print(df)
    
# check the centroid values by averaging the features in each cluster 
# es [cluster_label = 0, age = 33; cluster_label = 1, age = 50 ]
dataset_grouped_by_cluster = df.groupby('Cluster_label').mean()

if enablePlotDataset == True:
    print ("customer dataset grouped by cluster:\n")
    print(dataset_grouped_by_cluster)

# look at the distribution of customers based on their age and income
area = np.pi * ( X[:, 1])**2  

# scatter plot of age vs income with varying marker/color.
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5) # Age is feature 0; Income is 3 (CustomerId not considered)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()

# 3D
fig = plt.figure(1, figsize=(8, 6)) # create new figure
plt.clf() # clear figure
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134) # plot 3d axis on fig
plt.cla() # clear axis

# axis naming
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
# plot
ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float)) # education is col 1, Age is col 0, income is col 3

#
# Cluster group 
# EDUCATED-OLD 
# MIDDLE AGED-MIDDLE INCOME
# YOUNG-LOW INCOME
#
