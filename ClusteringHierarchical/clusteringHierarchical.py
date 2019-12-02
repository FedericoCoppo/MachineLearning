# author: Coppo Federico from Edx IBM course "Machine Learning with Python" 
# 02/11/2019 

# readme:
# waiting for 3.2.0 install the following release candidate
# $ pip install matplotlib==3.2.0rc1

# AGGLOMERATIVE HIERARCHICAL CLUSTERING (bottom up approach)

from matplotlib import pyplot as plt 
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets.samples_generator import make_blobs 
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# calibration
EnblRandomDataClustering = False

if EnblRandomDataClustering == True:
    # make_blobs class to generate random data: X1 is sample coordinates x y array, Y1 is integer labels for cluster membership of each sample
    X1, Y1 = make_blobs(n_samples=40, centers=[[2,1], [-2, -1.5], [1, -1], [-0.7,+1.9]], cluster_std=0.5)
    plt.scatter(X1[:, 0], X1[:, 1], marker='o') 
    centroids_num = 4

    # AVG LINKAGE CLUSTERING: distance for each cluster point to all cluster dataset
    clust_agglo = AgglomerativeClustering(n_clusters = centroids_num, linkage = 'average')  # create object
    clust_agglo.fit(X1,Y1) # method to fit the hierarchical clustering on the data

    # Create a figure
    f_size = [6.4, 4.8]
    plt.figure(figsize=f_size)

    # Create a minimum and maximum range of X1.
    x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0) # return the minimum and the max along X axes.

    # AVG distance for X1.
    X1 = (X1 - x_min) / (x_max - x_min) # X1max(1,1) X1min(0,0); normalization in [1x1 area]

    for i in range(X1.shape[0]): # X1 matrix row ->displays all of the datapoints.
        # Replace the data points with their respective cluster value 
        plt.text(X1[i, 0], X1[i, 1], str(Y1[i]),
                 color=plt.cm.nipy_spectral(clust_agglo.labels_[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 10})

    plt.scatter(X1[:, 0], X1[:, 1], marker='x') # Display original data before clustering
    plt.show() # Display plot

    # distance matrix
    dist_matrix = distance_matrix(X1,X1) 
    print(dist_matrix)

    # DENDOGRAM
    # linkage methods compute the distance between two clusters  
    linkage_matrix = hierarchy.linkage(dist_matrix, 'average') #UPGMA algo
    dendrogram = hierarchy.dendrogram(linkage_matrix) # bottom up dendogram

#
# Clustering on vehicle dataset: agglomerate similar car model 
#

# data read from csv
pdf = pd.read_csv('cars_clus.csv')
print ("dataset size (row, column): ", pdf.shape)

# clean row with null values
print ("Shape of dataset before cleaning (size):", pdf.size)
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
print ("Shape of dataset after cleaning (size): ", pdf.size)

# select the desidered features
featureSelection = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

# scale each feature to [0,1] range
x = featureSelection.values # it returns a numpy array
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)

# Clustering with Scipy
import scipy
import pylab
import scipy.cluster.hierarchy
from scipy.cluster.hierarchy import fcluster

lenght = feature_mtx.shape[0]
D = scipy.zeros([lenght,lenght])
for i in range(lenght):
    for j in range(lenght):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j]) # distance matrix

linkage_matrix = hierarchy.linkage(D, 'complete')
k = 5
clusters = fcluster(linkage_matrix, k, criterion='maxclust')

# plot dendogram
fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
dendro = hierarchy.dendrogram(linkage_matrix,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')

# Clustering with scikit-learn
dist_matrix = distance_matrix(feature_mtx,feature_mtx)
agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'average') # hierarchical clustering using bottom up approach
agglom.fit(feature_mtx)
pdf['cluster_'] = agglom.labels_ # new field to our dataframe to show the cluster of each row

# scatter plot distribution of each cluster  
import matplotlib.cm as cm
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
    subset = pdf[pdf.cluster_ == label]
    for i in subset.index:
            plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25) 
    plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)

plt.legend()
plt.title('CLUSTERS')
plt.xlabel('POWER')
plt.ylabel('MPG')
