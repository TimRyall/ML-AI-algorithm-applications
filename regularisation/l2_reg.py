import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

#adds the penalty term +λ||θ||^2_2 (where|| ||2 is the standard L2 norm, or Euclidean norm to our cost function.
#####################################################################
lambda_param = 0.01

data = pd.read_csv("linear-regression/pokemonregr.csv")
data = data.sample(frac=1) # shuffle data
data = data.fillna(0)

X = data.drop(columns=data.columns[-1])
#X = data.iloc[:, 0]
y = data.iloc[:, -1]

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Assign 70% training data 30% testing data for each data set
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3)

reg = Ridge(alpha = lambda_param)

reg.fit(XTrain, yTrain)

y_hat = reg.predict(XTest)

mse  = mean_squared_error(yTest, y_hat)
print('Slope (coef_): ', reg.coef_)
print('Intercept (intercept_): ', reg.intercept_)
print(f"Mean squared error: {mse}")
