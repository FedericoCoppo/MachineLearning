# author: Coppo Federico from Edx IBM course "Machine Learning with Python" 
# 03/11/2019 
# the script try to create a model to find out which drug might be appropriate for a future patient with the same illness

# Density-Based Spatial Clustering of Applications with Noise example

# Notice: For visualization of map, you need basemap package.
# if you dont have basemap install on your machine, you can use the following line to install it
# !conda install -c conda-forge  basemap==1.1.0  matplotlib==2.2.2  -y

import numpy as np 
from sklearn.cluster import DBSCAN 
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt

def createDataPoints(centroidLocation, numSamples, clusterDeviation):
    # Create random data and store in feature matrix X and response vector y.
    X, y = make_blobs(n_samples=numSamples, centers=centroidLocation, 
                                cluster_std=clusterDeviation)
    
    # Standardize features by removing the mean and scaling to unit variance
    X = StandardScaler().fit_transform(X)
    return X, y

# X is nx2 array of sample, y is array of cluster mebership of each sample
X, y = createDataPoints([[1,1], [0,1], [0,1.1]] , 400, 0.1)

# Instantiate the DBSCAN model
dbscan = DBSCAN(eps = 0.5, min_samples = 5)

# fit the model with data in array X.
model = dbscan.fit(X)

# print the predicted labels for each data point; same predicted labels represent the data points belonging to the same clusters.
# print(model.labels_)

# create an array of booleans using the labels from model.
core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
core_samples_mask[model.core_sample_indices_] = True

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
print("cluster num = %s" % n_clusters_ )

# remove repetition in labels by turning it into a set.
unique_labels = set(model.labels_) # sorted sequence of iterable elements

# data visualization
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels))) # colors creation.

# Plot the dataset
for k, col in zip(unique_labels, colors): #  zip takes in iterables as arguments and returns an iterator.
    if k == -1:
        col = 'k' # Black is noise.
    class_member_mask = (model.labels_ == k)

    # Plot the datapoints that are clustered
    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)

    # Plot the outliers
    xy = X[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker=u'o', alpha=0.5)

# COMPARISION WITH K-MEANS ALGO 
print("K-means algo plot:")
from sklearn.cluster import KMeans 
k = 3
k_means3 = KMeans(init = "k-means++", n_clusters = k, n_init = 12)
k_means3.fit(X)
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(k), colors):
    my_members = (k_means3.labels_ == k)
    plt.scatter(X[my_members, 0], X[my_members, 1],  c=col, marker=u'o', alpha=0.5)
plt.show()

#
# Weather Station dataset 
# Clustering using DBSCAN & scikit-learn EXAMPLE
#

import csv
import pandas as pd
import numpy as np

# load data
filename='weatherStations.csv'
pdf = pd.read_csv(filename)

# cleaning data
pdf = pdf[pd.notnull(pdf["Tm"])]   #remove rows that dont have any value (Tm field)
pdf = pdf.reset_index(drop=True) 

# data overview 
from mpl_toolkits.basemap import Basemap  # to plot plotting 2D data on maps 
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = (14,10)   
llon=-140                     # Columbia britanic
ulon=-50
llat=40
ulat=65

pdf = pdf[(pdf['Long'] > llon) & (pdf['Long'] < ulon) & (pdf['Lat'] > llat) &(pdf['Lat'] < ulat)]

my_map0 = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map0.drawcoastlines()
my_map0.drawcountries()
my_map0.fillcontinents(color = 'white', alpha = 0.3)
my_map0.shadedrelief()

# To collect data based on stations        
xs,ys = my_map0(np.asarray(pdf.Long), np.asarray(pdf.Lat))
pdf['xm']= xs.tolist() # long
pdf['ym'] =ys.tolist() # lat

#Visualization1
#for index,row in pdf.iterrows():
    #my_map0.plot(row.xm, row.ym,markerfacecolor =([1,0,0]),  marker='o', markersize= 5, alpha = 0.75)

# Clustering group of stations which show the same location (lat and long)

    # DBSCAN form sklearn library can runs DBSCAN clustering from vector array or distance matrix
    # we pass it the Numpy array Clus_dataSet to find core samples of high density and expands clusters from this sample
    
sklearn.utils.check_random_state(1000)
Clus_dataSet = pdf[['xm','ym']]
Clus_dataSet = np.nan_to_num(Clus_dataSet)
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

# Compute DBSCAN
db = DBSCAN(eps=0.15, min_samples=10).fit(Clus_dataSet)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
pdf["Clus_Db"]=labels

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels)) 
# print(clusterNum) #6

# A sample of clusters
pdf[["Stn_Name","Tx","Tm","Clus_Db"]].head(5) 

# display the "location" clustering result  (using basemap)
rcParams['figure.figsize'] = (14,10)

my_map1 = Basemap(projection='merc',
            resolution = 'l', area_thresh = 1000.0,
            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)
            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)

my_map1.drawcoastlines()
my_map1.drawcountries()
#my_map1.drawmapboundary()
my_map1.fillcontinents(color = 'white', alpha = 0.3)
my_map1.shadedrelief()

# To create a color map
colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))

#Visualization1
for clust_number in set(labels):
    c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
    clust_set = pdf[pdf.Clus_Db == clust_number]                    
    my_map1.scatter(clust_set.xm, clust_set.ym, color =c,  marker='o', s= 20, alpha = 0.85)
    if clust_number != -1:
        cenx=np.mean(clust_set.xm) 
        ceny=np.mean(clust_set.ym) 
        plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)
        print ("Cluster "+str(clust_number)+', Avg Temp: '+ str(np.mean(clust_set.Tm)))

# Clustering of stations based on their location, mean, max, and min Temperature
Clus_dataSet = pdf[['xm','ym','Tx','Tm','Tn']]
Clus_dataSet = np.nan_to_num(Clus_dataSet)
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(Clus_dataSet)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
pdf["Clus_Db"]=labels

realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
clusterNum = len(set(labels)) 

# A sample of clusters
pdf[["Stn_Name","Tx","Tm","Clus_Db"]].head(5)

print ("end of script!")
