import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

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
    data = pd.read_csv("4.1-linear-regression/pokemonregr.csv")
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

    reg = LinearRegression()

    reg.fit(XTrain, yTrain)

    y_hat = reg.predict(XTest)

    #plt.plot(x, y, color='blue', label='True function')
    #plt.plot(XTrain, y_hat, color='green', label='Regression function')
    #plt.scatter(XTrain, yTrain, color='red', label='Training data')
    #plt.legend()
    #plt.show()

    mse  = mean_squared_error(yTest, y_hat)
    print('Slope (coef_): ', reg.coef_)
    print('Intercept (intercept_): ', reg.intercept_)
    print(f"Mean squared error: {mse}")



##################################################
##################################################

poke_regr()


##################################################
##################################################