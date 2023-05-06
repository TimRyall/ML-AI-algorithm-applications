import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# Import classifcation data
classifDf = pd.read_csv("9.1-bagging/classif.csv")

# Assign the inputs (1st & 2nd col of df) and output (3rd col)
X = classifDf.drop(columns=classifDf.columns[-1])
y = classifDf.iloc[:, -1]

# Assign 70% training data 30% testing data for each data set
X_train, X_ho, y_train, y_ho = train_test_split(X, y, test_size=0.3)

# Create RF model with depth specifed
depths = [i for i in range(1,1000, 50)]
hos = []
for depth in depths:
    rf_model = RandomForestClassifier(max_depth=3, n_estimators=depth, random_state=42)
    rf_model.fit(X_train, y_train)

    # Use the model to predcit classes based on the test inputs
    y_pred_ho = rf_model.predict(X_ho)
    y_pred_tr = rf_model.predict(X_train)

    # Compare the predicted classes with the real test classes
    # accuracy = % of correct predictions 
    # missclass = % of incorrect predictions
    misclassRateHO  = 1 - accuracy_score(y_ho, y_pred_ho)
    '''
    misclassRateTr  = 1 - accuracy_score(y_train, y_pred_tr)
    print(f"E_train (Misclassification rate of DT) (depth = {depth}): {misclassRateTr}\n")
    misclassRateHO  = 1 - accuracy_score(y_ho, y_pred_ho)
    print(f"E_ho (Misclassification rate of DT) (depth = {depth}): {misclassRateHO}\n")
    '''
    hos.append(misclassRateHO)

plt.plot(depths, hos)
plt.xlabel('depth')
plt.ylabel('misclassifcation rate')
plt.title("misclassifcation rates for diffrent depths")
plt.show()