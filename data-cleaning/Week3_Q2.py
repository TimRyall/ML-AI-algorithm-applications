import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

###############################################################################
# Q2
###############################################################################

# Import regression data
regrDf = pd.read_csv("Week3/w3regr.csv")
# Import classifcation data
classifDf = pd.read_csv("Week3/w3classif.csv")
#print("OLD data before Shuffling: \n")
#print(classifDf)

# Shuffle the data randomly 
# frac=1 returns 100% of the data in the "random sample"
regrDf = regrDf.sample(frac=1) 
classifDf = classifDf.sample(frac=1) 
#print("NEW data after Shuffling: \n")
#print(classifDf)

# regression
# Assign the input (first col of df) and output (second col)
inputRegr = regrDf.drop(columns=regrDf.columns[1])
outputRegr = regrDf.iloc[:, 1]

# classifcation
# Assign the inputs (1st & 2nd col of df) and output (3rd col)
inputClassif= classifDf.drop(columns=classifDf.columns[2])
outputClassif = classifDf.columns[2]

# Assign 70% training data 30% testing data for each data set
XTrainRegr, XTestRegr, yTrainRegr, yTestRegr = train_test_split(
    inputRegr, outputRegr, test_size=0.3)
XTrainClassif, XTestClassif, yTrainClassif, yTestClassif = train_test_split(
    inputRegr, outputRegr, test_size=0.3)