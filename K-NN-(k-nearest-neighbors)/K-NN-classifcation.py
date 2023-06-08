<<<<<<< HEAD:3.2-K-NN-(k-nearest-neighbors)/K-NN-classifcation.py
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

###############################################################################

# Import classifcation data
classifDf = pd.read_csv("classif.csv", names=["x1","x2","y"])

# Shuffle the data randomly 
# frac=1 returns 100% of the data in the "random sample"
classifDf = classifDf.sample(frac=1) 

# classifcation
# Assign the inputs (1st & 2nd col of df) and output (3rd col)
inputClassif= classifDf.drop(columns=classifDf.columns[2])
outputClassif = classifDf.iloc[:, 2]

# Assign 70% training data 30% testing data for each data set
XTrainClassif, XTestClassif, yTrainClassif, yTestClassif = train_test_split(
    inputClassif, outputClassif, test_size=0.3)

# Create K-NN model with k = 3 for our classifcation data
k = 3
knnClassif = KNeighborsClassifier(n_neighbors=k)

# Train the model with our training data
knnClassif.fit(XTrainClassif, yTrainClassif)
# Use the model to predcit classes based on the test inputs
yPredClassif = knnClassif.predict(XTestClassif)

# Compare the predicted classes with the real test classes
# accuracy = % of correct predictions 
# missclass = $ of incorrect predictions
misclassRate  = 1 - accuracy_score(yTestClassif, yPredClassif)
print(f"Misclassification rate  of k-NN (k = {k}): {misclassRate}")

###############################################################################

# Define mesh grid inorder to draw decision regions
# size of mesh grid is large enough to fit all data points
x1, x2 = np.meshgrid(np.arange(np.min(inputClassif.iloc[:, 0])-2, 
                               np.max(inputClassif.iloc[:, 0])+2, 0.01)
                     , np.arange(np.min(inputClassif.iloc[:, 1])-2, 
                                 np.max(inputClassif.iloc[:, 1])+2, 0.01))

# Predict class labels for each point in mesh grid
# ravel() flattens a multi-dimensional array into a 1-D array.
# np.c_ concatenates two 1-D arrays along the second axis to create a 2-D array.
Y = knnClassif.predict(np.c_[x1.ravel(), x2.ravel()])
# Reshape predicted class labels to the shape of the mesh grid
Y = Y.reshape(x1.shape)

# Plot decision regions and scatter plot original data
plt.contourf(x1, x2, Y, alpha=0.4)
plt.scatter(inputClassif.iloc[:, 0], inputClassif.iloc[:, 1], c=outputClassif, alpha=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'KNN Decision Regions for k={k}')
plt.show()
# This will produce a scatter plot with the decision regions for the KNN classifier.

###############################################################################

# create array to store misclassifcation rates
misclassRates = []
for k in range(1,20):
    # Create K-NN model with k for our classifcation data
    knnClassif = KNeighborsClassifier(n_neighbors=k)

    # Train the model with our training data
    knnClassif.fit(XTrainClassif, yTrainClassif)
    # Use the model to predcit classes based on the test inputs
    yPredClassif = knnClassif.predict(XTestClassif)

    # Compare the predicted classes with the real test classes
    # accuracy = % of correct predictions 
    # missclass = $ of incorrect predictions
    misclassRate  = 1 - accuracy_score(yTestClassif, yPredClassif)
    misclassRates.append(misclassRate)

plt.scatter(range(1,20), misclassRates)
plt.xlabel('k')
plt.ylabel('misclassifcation rate')
plt.title("misclassifcation rates for diffrent k")
plt.show()
=======
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Build a k-NN classifier with k = 3 for dataset classif.csv and find the training
# and test loss (i.e. misclassification rate).
###############################################################################

# Import classifcation data
classifDf = pd.read_csv("classif.csv", names=["x1","x2","y"])

# Shuffle the data randomly 
# frac=1 returns 100% of the data in the "random sample"
classifDf = classifDf.sample(frac=1) 

# classifcation
# Assign the inputs (1st & 2nd col of df) and output (3rd col)
inputClassif= classifDf.drop(columns=classifDf.columns[2])
outputClassif = classifDf.iloc[:, 2]

# Assign 70% training data 30% testing data for each data set
XTrainClassif, XTestClassif, yTrainClassif, yTestClassif = train_test_split(
    inputClassif, outputClassif, test_size=0.3)

# Create K-NN model with k = 3 for our classifcation data
k = 3
knnClassif = KNeighborsClassifier(n_neighbors=k)

# Train the model with our training data
knnClassif.fit(XTrainClassif, yTrainClassif)
# Use the model to predcit classes based on the test inputs
yPredClassif = knnClassif.predict(XTestClassif)

# Compare the predicted classes with the real test classes
# accuracy = % of correct predictions 
# missclass = $ of incorrect predictions
misclassRate  = 1 - accuracy_score(yTestClassif, yPredClassif)
print(f"Misclassification rate  of k-NN (k = {k}): {misclassRate}")

###############################################################################
# Q3 b)
# Plot the decision regions for your classifier together with the training and/or test
# data points.
###############################################################################

# Define mesh grid inorder to draw decision regions
# size of mesh grid is large enough to fit all data points
x1, x2 = np.meshgrid(np.arange(np.min(inputClassif.iloc[:, 0])-2, 
                               np.max(inputClassif.iloc[:, 0])+2, 0.01)
                     , np.arange(np.min(inputClassif.iloc[:, 1])-2, 
                                 np.max(inputClassif.iloc[:, 1])+2, 0.01))

# Predict class labels for each point in mesh grid
# ravel() flattens a multi-dimensional array into a 1-D array.
# np.c_ concatenates two 1-D arrays along the second axis to create a 2-D array.
Y = knnClassif.predict(np.c_[x1.ravel(), x2.ravel()])
# Reshape predicted class labels to the shape of the mesh grid
Y = Y.reshape(x1.shape)

# Plot decision regions and scatter plot original data
plt.contourf(x1, x2, Y, alpha=0.4)
plt.scatter(inputClassif.iloc[:, 0], inputClassif.iloc[:, 1], c=outputClassif, alpha=0.8)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'KNN Decision Regions for k={k}')
plt.show()
# This will produce a scatter plot with the decision regions for the KNN classifier.

###############################################################################
# Experiment with different k values and see how it affects the loss values and the
# decision regions.
###############################################################################

# create array to store misclassifcation rates
misclassRates = []
for k in range(1,20):
    # Create K-NN model with k for our classifcation data
    knnClassif = KNeighborsClassifier(n_neighbors=k)

    # Train the model with our training data
    knnClassif.fit(XTrainClassif, yTrainClassif)
    # Use the model to predcit classes based on the test inputs
    yPredClassif = knnClassif.predict(XTestClassif)

    # Compare the predicted classes with the real test classes
    # accuracy = % of correct predictions 
    # missclass = $ of incorrect predictions
    misclassRate  = 1 - accuracy_score(yTestClassif, yPredClassif)
    misclassRates.append(misclassRate)

plt.scatter(range(1,20), misclassRates)
plt.xlabel('k')
plt.ylabel('misclassifcation rate')
plt.title("misclassifcation rates for diffrent k")
plt.show()
>>>>>>> d6f486078cf5418ccacd447c1afa7364338516b4:K-NN-(k-nearest-neighbors)/K-NN-classifcation.py
