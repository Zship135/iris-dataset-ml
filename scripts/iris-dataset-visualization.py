#==================================================================#
# Visualization of Iris Dataset                                    #
#==================================================================#
# Purpose:
#   To make a pairplot of petal length, petal width, sepal length, and sepal width to gain an understanding of the correlation between features.

# IMPORTS #
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# LOAD DATA SET AND CONVERT TO DATA FRAME #
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# VISUALIZATION #
selected_features = ['petal length (cm)', 'petal width (cm)', 'sepal length (cm)', 'sepal width (cm)', 'target_name']
sns.pairplot(df[selected_features], hue='target_name', palette='Set1')
plt.suptitle('Feature Relationships in the Iris Dataset')
plt.show()
