import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm # support vector machines
from sklearn.metrics import accuracy_score

# load in the data
df = pd.read_csv("support-vector-machines/diabetes.csv")
df.columns.values[-1] = "prediction"

# last column is output
y = df.iloc[:,-1]
X = df.iloc[:,:-1]

# split data in to training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create and train SVM model
reg_param = 1

model = svm.SVC(kernel='linear', C=reg_param)
model.fit(X_train, y_train)

# predict the test data
y_pred = model.predict(X_test)

# report the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy,3))