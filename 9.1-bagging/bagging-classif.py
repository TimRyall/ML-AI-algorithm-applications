import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Import classifcation data
classifDf = pd.read_csv("9.1-bagging/classif.csv")

# Assign the inputs (1st & 2nd col of df) and output (3rd col)
X = classifDf.drop(columns=classifDf.columns[-1])
y = classifDf.iloc[:, -1]

# Assign 70% training data 30% testing data for each data set
X_train, X_ho, y_train, y_ho = train_test_split(X, y, test_size=0.3)

# Create DT model with depth specifed
depth = 3
DTClassif = DecisionTreeClassifier(max_depth=depth)

# Train the model with our training data
DTClassif.fit(X_train, y_train)
# Use the model to predcit classes based on the test inputs
y_pred_ho = DTClassif.predict(X_ho)
y_pred_tr = DTClassif.predict(X_train)

# Compare the predicted classes with the real test classes
# accuracy = % of correct predictions 
# missclass = % of incorrect predictions
misclassRateTr  = 1 - accuracy_score(y_train, y_pred_tr)
print(f"E_train (Misclassification rate of DT) (depth = {depth}): {misclassRateTr}\n")
misclassRateHO  = 1 - accuracy_score(y_ho, y_pred_ho)
print(f"E_ho (Misclassification rate of DT) (depth = {depth}): {misclassRateHO}\n")
