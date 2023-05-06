import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

import matplotlib.patches as mpatches

#####################################################################

# Define the function
def f(x):
    return x**3 + 1

def draw_true_func():
    # Define the domain
    x = np.linspace(-1, 1, 100)
    # Compute the function values
    y = f(x)
    # Plot the function
    plt.plot(x, y)
    # Add labels and title
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Plot of f(x) = x^3 + 1')
    # Show the plot
    plt.show()

#####################################################################

def poke_regr():
    data = pd.read_csv("4.1-linear-regression/classif.csv")
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

    # Instantiate logistic regression model
    logreg = LogisticRegression()

    # Fit the model to the training data
    logreg.fit(XTrain, yTrain)

    yhat = logreg.predict(XTest)

    inputd = [[1.1, 1.1]]
    print(logreg.predict_proba(inputd))

    #plt.plot(x, y, color='blue', label='True function')
    #plt.plot(XTrain, y_hat, color='green', label='Regression function')
    #plt.scatter(XTrain, yTrain, color='red', label='Training data')
    #plt.legend()
    #plt.show()

    print('Slope (coef_): ', logreg.coef_)
    print('Intercept (intercept_): ', logreg.intercept_)
    print('Accuracy:', accuracy_score(yTest, yhat))
    print('Confusion matrix:\n', confusion_matrix(yTest, yhat))

    # Define a meshgrid of points over which we will evaluate the model's predictions
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Get the model's predicted probabilities for each point on the meshgrid
    Z = logreg.predict_proba(grid)[:, 1]
    Z = Z.reshape(xx.shape)
    print(Z)

    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y)

    # Plot the decision regions
    plt.contourf(xx, yy, Z, alpha=0.4)

    # Plot the decision boundary as a hard line
    w = logreg.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(x_min+1, x_max-1)
    yy = a * xx - (logreg.intercept_[0]) / w[1]
    plt.plot(xx, yy, 'k-')

    plt.show()

##################################################
##################################################

poke_regr()


##################################################
##################################################