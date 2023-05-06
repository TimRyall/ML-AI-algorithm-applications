import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

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

def plot_train_data():
    # Generate 30 random input values between -1 and 1
    x_train = np.random.uniform(low=-1, high=1, size=(30,))

    # Evaluate the true function at each input value and add Gaussian noise
    y_train = f(x_train) + np.random.normal(loc=0, scale=0.1, size=(30,))

    # Plot the function and training data
    x = np.linspace(-1, 1, 100)
    y = f(x)

    plt.plot(x, y, color='blue', label='True function')
    plt.scatter(x_train, y_train, color='red', label='Training data')
    plt.legend()
    plt.show()


#########################################################################

def lin_reg_first_order_poly():
    # Generate 30 random input values between -1 and 1
    X_train = np.random.uniform(low=-1, high=1, size=(30,))

    # Evaluate the true function at each input value and add Gaussian noise
    y_train = f(X_train) + np.random.normal(loc=0, scale=0.1, size=(30,))
    y_train = y_train.reshape(-1, 1)
    X_train = X_train.reshape(-1, 1)

    reg = LinearRegression()

    reg.fit(X_train, y_train)

    # Plot graphs
    x = np.linspace(-1, 1, 100)
    y = f(x)
    y_hat = reg.predict(X_train)

    plt.plot(x, y, color='blue', label='True function')
    plt.plot(X_train, y_hat, color='green', label='Regression function')
    plt.scatter(X_train, y_train, color='red', label='Training data')
    plt.legend()
    plt.show()


#########################################################################

def lin_reg_second_order_poly():
    # Generate 30 random input values between -1 and 1
    X_train = np.random.uniform(low=-1, high=1, size=(30,))
    X_train = np.sort(X_train) 

    # Evaluate the true function at each input value and add Gaussian noise
    y_train = f(X_train) + np.random.normal(loc=0, scale=0.1, size=(30,))
    y_train = y_train.reshape(-1, 1)
    X_train = X_train.reshape(-1, 1)

    X_new = np.concatenate((X_train, np.power(X_train,2)), axis=1)

    reg = LinearRegression()

    reg.fit(X_new, y_train)

    # Plot graphs
    x = np.linspace(-1, 1, 100)
    y = f(x)
    y_hat = reg.predict(X_new)


    plt.plot(x, y, color='blue', label='True function')
    plt.plot(X_new[:, 0], y_hat, color='green', label='Regression function')
    plt.scatter(X_train, y_train, color='red', label='Training data')
    plt.legend()
    plt.show()

def lin_reg_third_order_poly():
    # Generate 30 random input values between -1 and 1
    X_train = np.random.uniform(low=-1, high=1, size=(30,))
    X_train = np.sort(X_train) 

    # Evaluate the true function at each input value and add Gaussian noise
    y_train = f(X_train) + np.random.normal(loc=0, scale=0.1, size=(30,))
    y_train = y_train.reshape(-1, 1)
    X_train = X_train.reshape(-1, 1)

    X_new = np.concatenate((X_train, np.power(X_train,2)), axis=1)
    X_new = np.concatenate((X_new, np.power(X_train,3)), axis=1)

    reg = LinearRegression()

    reg.fit(X_new, y_train)

    # Plot graphs
    x = np.linspace(-1, 1, 100)
    y = f(x)
    y_hat = reg.predict(X_new)


    plt.plot(x, y, color='blue', label='True function')
    plt.plot(X_new[:, 0], y_hat, color='green', label='Regression function')
    plt.scatter(X_train, y_train, color='red', label='Training data')
    plt.legend()
    plt.show()

##################################################
##################################################

lin_reg_third_order_poly()


##################################################
##################################################