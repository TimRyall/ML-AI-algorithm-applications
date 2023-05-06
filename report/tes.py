import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt

# Data for the bars
x = ['Simple linear', 'Multiple Linear', 'Simple Polynomial', 'Multiple Polynomial']
y = [124510.10, 18925.41, 118422.08, 7028.20]

# Create a bar chart
plt.bar(x, y, color='darkseagreen')

for i in range(len(x)):
    plt.text(i, y[i], y[i], ha='center', va='bottom')

# Add a title and labels for the axes
plt.title('MSE Comparision of our diffrent regession models')
plt.xlabel('Model used')
plt.ylabel('Average MSE')

# Rotate the x-axis labels
plt.xticks(rotation=45)


# Adjust the margins to prevent the plot from being cut off
plt.savefig('reg_comparison.png', bbox_inches='tight')
plt.show()
