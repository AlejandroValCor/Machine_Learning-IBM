# k-Means on a randomly generated dataset
# Importing needed packages
# Let's first import the required libraries. Also run %matplotlib inline since we will be plotting in this section.

# await removed due to only use inside async function error
#you are running the lab in your  browser, so we will install the libraries using ``piplite``
import piplite
piplite.install(['pandas'])
piplite.install(['matplotlib'])
piplite.install(['scipy'])
piplite.install(['seaborn'])

import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
%matplotlib inline

# download the data
from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

# k-Means on a randomly generated dataset
# Let's create our own dataset for this lab!
# First we need to set a random seed. Use numpy's random.seed() function, where the seed will be set to 0.
np.random.seed(0)
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
plt.scatter(X[:, 0], X[:, 1], marker='.')

# Setting up K-Means
# The KMeans class has many parameters that can be used, but we will be using these three:

# init: Initialization method of the centroids.
# Value will be: "k-means++"
# k-means++: Selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
# n_clusters: The number of clusters to form as well as the number of centroids to generate.
# Value will be: 4 (since we have 4 centers)
# n_init: Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
# Value will be: 12
# Initialize KMeans with these parameters, where the output parameter is called k_means.

k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)

# Now let's fit the KMeans model with the feature matrix we created above, X 
k_means.fit(X)

# Now let's grab the labels for each point in the model using KMeans' .labels_ attribute and save it as k_means_labels .
k_means_labels = k_means.labels_
k_means_labels

# We will also get the coordinates of the cluster centers using KMeans' .cluster_centers_ and save it as k_means_cluster_centers .
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers

# Creating the Visual Plot
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()

# Practice
# Try to cluster the above dataset into 3 clusters.
k_means3 = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)
k_means3.fit(X)
fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means3.labels_))))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len(k_means3.cluster_centers_)), colors):
    my_members = (k_means3.labels_ == k)
    cluster_center = k_means3.cluster_centers_[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
plt.show()

# Customer Segmentation with K-Means
# Let's download the datasetfrom IBM Object Storage.
path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv'

# Load Data From CSV File
await download(path, "Cust_Segmentation.csv")
filename ="Cust_Segmentation.csv"

import pandas as pd
cust_df = pd.read_csv("Cust_Segmentation.csv")
cust_df.head()

# Pre-processing
# As you can see, Address in this dataset is a categorical variable. The k-means algorithm isn't directly applicable to categorical variables
# because the Euclidean distance function isn't really meaningful for discrete variables. So, let's drop this feature and run clustering.
df = cust_df.drop('Address', axis=1)
df.head()

# Normalizing over the standard deviation
# Now let's normalize the dataset. But why do we need normalization in the first place? Normalization is a statistical method that helps
# mathematical-based algorithms to interpret features with different magnitudes and distributions equally. We use StandardScaler() to normalize
# our dataset.
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet

# Modeling
# Let's apply k-means on our dataset, and take a look at cluster labels.
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

# Insights
# We assign the labels to each row in the dataframe.
df["Clus_km"] = labels
df.head(5)

# We can easily check the centroid values by averaging the features in each cluster.
df.groupby('Clus_km').mean()

# Now, let's look at the distribution of customers based on their age and income:
area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()

from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))

# k-means will partition your customers into mutually exclusive groups, for example, into 3 clusters. The customers in each cluster are similar
# to each other demographically. Now we can create a profile for each group, considering the common characteristics of each cluster. For
# example, the 3 clusters can be:

# AFFLUENT, EDUCATED AND OLD AGED
# MIDDLE AGED AND MIDDLE INCOME
# YOUNG AND LOW INCOME