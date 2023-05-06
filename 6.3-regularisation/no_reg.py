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
# In Prac W4 we applied linear regression to a pokemon dataset, where the
# loss function was sum of squares (or mean squared) error. Revisit this task but add (a)
# L2; (b) L1 regularisation to the loss function, with some suitable value for the
# regularization hyperparameter (see Section 5.3 of the textbook). Compare the
# coefficient values from your different trained models.
#
######################################################################

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

    mse  = mean_squared_error(yTest, y_hat)
    print('Slope (coef_): ', reg.coef_)
    print('Intercept (intercept_): ', reg.intercept_)
    print(f"Mean squared error: {mse}")



##################################################
##################################################

poke_regr()


##################################################
##################################################