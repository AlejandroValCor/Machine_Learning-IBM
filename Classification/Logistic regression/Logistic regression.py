# Customer churn with Logistic Regression

# Importing needed packages
# await removed due to only use inside async function error
# !pip install scikit-learn==0.23.1
import piplite
piplite.install(['pandas'])
piplite.install(['matplotlib'])
piplite.install(['numpy'])
piplite.install(['scikit-learn'])
piplite.install(['scipy'])

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
%matplotlib inline 
import matplotlib.pyplot as plt

from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

# Load the Telco Churn data
#Click here and press Shift+Enter
path= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"

# Load data from csv file
# await removed due to only use inside async function error
download(path, "ChurnData.csv")
path="ChurnData.csv"

churn_df = pd.read_csv(path)
churn_df.head()

# Data pre-processing and selection
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df.head()

# Practice
# How many rows and columns are in this dataset in total? What are the names of columns?
churn_df.shape

# Let's define X, and y for our dataset:
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]
y = np.asarray(churn_df['churn'])
y [0:5]

# Also, we normalize the dataset:

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

# Train/Test dataset
# We split our dataset into train and test set:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# Modeling (Logistic Regression with Scikit-learn)
# Let's build our model using LogisticRegression from the Scikit-learn package. This function implements logistic regression and can use 
# different numerical optimizers to find parameters, including ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’ solvers. You can find extensive
# information about the pros and cons of these optimizers if you search it in the internet.

# The version of Logistic Regression in Scikit-learn, support regularization. Regularization is a technique used to solve the overfitting
# problem of machine learning models. C parameter indicates inverse of regularization strength which must be a positive float. Smaller values
# specify stronger regularization. Now let's fit our model with train set:
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR

# Now we can predict using our test set:
yhat = LR.predict(X_test)
yhat

# predict_proba returns estimates for all classes, ordered by the label of classes. So, the first column is the probability of class 0, P(Y=0|X),
# and second column is probability of class 1, P(Y=1|X):
yhat_prob = LR.predict_proba(X_test)
yhat_prob

# Evaluation
# jaccard index
# Let's try the jaccard index for accuracy evaluation. we can define jaccard as the size of the intersection divided by the size of the union of
# the two label sets. If the entire set of predicted labels for a sample strictly matches with the true set of labels, then the subset accuracy
# is 1.0; otherwise it is 0.0.
from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat,pos_label=0)

# confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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
print(confusion_matrix(y_test, yhat, labels=[1,0]))
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

print (classification_report(y_test, yhat))

# Based on the count of each section, we can calculate precision and recall of each label:

# Precision is a measure of the accuracy provided that a class label has been predicted. It is defined by: precision = TP / (TP + FP)
# Recall is the true positive rate. It is defined as: Recall =  TP / (TP + FN)

# So, we can calculate the precision and recall of each class.
# F1 score: Now we are in the position to calculate the F1 scores for each label based on the precision and recall of that label.

# The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall)
# and worst at 0. It is a good way to show that a classifer has a good value for both recall and precision.
# Finally, we can tell the average accuracy for this classifier is the average of the F1-score for both labels, which is 0.72 in our case.

# log loss
# Now, let's try log loss for evaluation. In logistic regression, the output can be the probability of customer churn is yes (or equals to 1).
# This probability is a value between 0 and 1. Log loss( Logarithmic loss) measures the performance of a classifier where the predicted output
# is a probability value between 0 and 1.
from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)

# Practice
# Try to build Logistic Regression model again for the same dataset, but this time, use different __solver__ and __regularization__ values?
# What is new __logLoss__ value?
LR2 = LogisticRegression(C=0.0001, solver='sag').fit(X_train,y_train)
yhat_prob2 = LR2.predict_proba(X_test)
print ("LogLoss: : %.2f" % log_loss(y_test, yhat_prob2))