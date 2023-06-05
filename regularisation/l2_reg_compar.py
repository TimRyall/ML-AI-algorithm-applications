import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

#####################################################################

data = pd.read_csv("linear-regression/pokemonregr.csv")
data = data.sample(frac=1) # shuffle data
data = data.fillna(0)

X = data.drop(columns=data.columns[-1])
y = data.iloc[:, -1]

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Assign 70% training data 30% testing data for each data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

lambdas = np.logspace(-4, 4, num=20)
test_mse_per_lambda = []
train_mse_per_lambda = []
for lambda_l1 in lambdas:
    test_current_lambda_mse = []
    train_current_lambda_mse = []
    for trial in range(1):
        # train and fit the linear regression model
        reg = Ridge(alpha=lambda_l1, max_iter=10000) # alpha is our regularisation param
        reg.fit(X_train, y_train)

        # apply the linear model to our test data
        y_hat_test = reg.predict(X_test)
        y_hat_train = reg.predict(X_train)

        test_mse = mean_squared_error(y_test, y_hat_test) # calculate MSE
        train_mse = mean_squared_error(y_train, y_hat_train)

        test_current_lambda_mse.append(test_mse) # add mse to lsit
        train_current_lambda_mse.append(train_mse)
    #print(f'lambda: {lambda_l1}: {np.mean(current_lambda_mse)}') # average MSE
    
    test_mse_per_lambda.append(np.mean(test_current_lambda_mse))
    train_mse_per_lambda.append(np.mean(train_current_lambda_mse))


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 8))
ax1.loglog(lambdas, test_mse_per_lambda, color='lightcoral', label='Testing Error')
ax1.loglog(lambdas, train_mse_per_lambda, color='darkseagreen', label='Training Error')
ax1.set_title('Training and testing error for model \n using L2 regularisation')
ax1.set_xlabel('Regularisation parameter (lambda) \n Model complexity increases ->')
ax1.set_ylabel('Error (MSE)')
ax1.legend()
ax1.invert_xaxis()

g_gap = result = [b - a for a, b in zip(train_mse_per_lambda, test_mse_per_lambda)]

ax2.loglog(lambdas, g_gap, color='gray')
ax2.set_title(f'Generalisation Gap')
ax2.set_xlabel(f'Regularisation parameter (lambda) \n Model complexity increases ->')
ax2.set_ylabel('Generalisation Gap (Etrain - Etest)')
plt.xlim(min(lambdas)-1, max(lambdas)+1)
plt.ylim(min(g_gap)-1, max(g_gap)+1)

ax2.invert_xaxis()



#plt.savefig(f'{input_feature}.png', bbox_inches='tight')
plt.show()
