import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Import regression data
regrDf = pd.read_csv("Week3/w3regr.csv")
# Assign the input (first col of df) and output (second col)
inputRegr = regrDf.iloc[:, 0]
outputRegr = regrDf.iloc[:, 1]

# Import classifcation data
classifDf = pd.read_csv("Week3/w3classif.csv")
# Assign the inputs (1st & 2nd col of df) and output (3rd col)
input1Classif = classifDf.iloc[:, 0]
input2Classif = classifDf.iloc[:, 1]
outputClassif = classifDf.iloc[:, 2]

# Create two subplots
fig, (regrAxi, classifAxi) = plt.subplots(1, 2, figsize=(10, 5))

# Create regression scatter plot
regrAxi.scatter(inputRegr, outputRegr)
regrAxi.set_title('Regression data')
regrAxi.set_xlabel('Input')
regrAxi.set_ylabel('Output')

# Create classification scatter plot
# Colours denote the different classes
classifPlt = classifAxi.scatter(input1Classif, input2Classif, 
                   c=outputClassif)
classifAxi.set_title('Classification data')
classifAxi.set_xlabel('Input 1')
classifAxi.set_ylabel('Input 2')
# Display ledgend for the class colours
legend1 = classifAxi.legend(*classifPlt.legend_elements(),
                    loc="lower right", title="Classes")
classifAxi.add_artist(legend1)

# Display the plots here
plt.show()

