# Importing needed packages
#!pip install scikit-learn==0.23.1

# await removed due to only use inside async function error
import piplite
piplite.install(['pandas'])
piplite.install(['matplotlib'])
piplite.install(['numpy'])
piplite.install(['scikit-learn'])
piplite.install(['scipy'])

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
%matplotlib inline

# Download data set
path="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv"

from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

# Load data from csv file
# await removed due to only use inside async function error
download(path, 'teleCust1000t.csv')
df = pd.read_csv('teleCust1000t.csv')
df.head()

# Data Visualization and Analysis
# Let’s see how many of each class is in our data set
df['custcat'].value_counts()

# You can easily explore your data using visualization techniques:
df.hist(column='income', bins=50)

# Features sets
# Let's define features sets X.
df.columns

# To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]

# What are our labels?
y = df['custcat'].values
y[0:5]

# Normalize data
# Data Standardization gives the data zero mean and unit variance, it is good practice, especially for algorithms such as KNN which is based
# on the distance of data points:

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# Classification
# Import classifier library implementing k-neasrest neighbors vote.
from sklearn.neighbors import KNeighborsClassifier

# Training
# Let's start the algorithm with k=4 for now
k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

# Predicting
yhat = neigh.predict(X_test)
yhat[0:5]

# Accuracy evaluation
# In multilabel classification, **accuracy classification score** is a function that computes subset accuracy. This function is equal to the
# jaccard_score function. Essentially, it calculates how closely the actual labels and predicted labels are matched in the test set.

from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

# Practice
# Can you build the model but with k =6?
k = 6
neigh6 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat6 = neigh6.predict(X_test)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh6.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat6))

# What about other k?
# K in KNN, is the number of nearest neighbors to examine. It is supposed to be specified by the user. So, how can we choose right value for K?
# The general solution is to reserve a part of your data for testing the accuracy of the model. Then choose k =1, use the training part for
# modeling, and calculate the accuracy of prediction using all samples in your test set. Repeat this process, increasing the k, and see which
# k is the best for your model.

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

# Plot the model accuracy for a different number of neighbors.
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 