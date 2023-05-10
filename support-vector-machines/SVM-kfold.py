import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn import svm # support vector machines
from sklearn.metrics import accuracy_score

# load in the data
df = pd.read_csv("support-vector-machines/diabetes.csv")
df.columns.values[-1] = "prediction"

# last column is output
y = df.iloc[:,-1]
X = df.iloc[:,:-1]

# create SVM model
reg_param = 0.1
model = svm.SVC(kernel='linear', C=reg_param)

# create a KFold object with K=3
k = 3
kf = KFold(n_splits=k, shuffle=True)

# create list to store the accuarcy for each fold
accuracy_scores = []

# perform cross-validation
for train_index, test_index in kf.split(X):
    # assign train (k-1/k) of data points
    # assign test (1/k) of data points
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # fit the model with the training data
    model.fit(X_train, y_train)

    # predict the test data
    y_pred = model.predict(X_test)

    # compute accuracy for this FOLD
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

print("Accuracy:", round(np.mean(accuracy_scores), 4))
print("Variance:", round(np.var(accuracy_scores), 4))


