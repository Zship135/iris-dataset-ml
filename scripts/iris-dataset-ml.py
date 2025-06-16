#==================================================================#
# Visualization of Iris Dataset                                    #
#==================================================================#
# Purpose:
#   To make a pairplot of petal length, petal width, sepal length, and sepal width to gain an understanding of the correlation between features.

# IMPORTS #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# LOAD DATA SET AND CONVERT TO DATA FRAME #
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# EDA #
print(df.info())
print(df.describe())
print(df['target_name'].value_counts())

# VISUALIZATION #
selected_features = ['petal length (cm)', 'petal width (cm)', 'sepal length (cm)', 'sepal width (cm)', 'target_name']
sns.pairplot(df[selected_features], hue='target_name', palette='Set1')
plt.suptitle('Feature Relationships in the Iris Dataset')
plt.show()

# TRAIN MODEL #
# LOGISTIC REGRESSION #
x = df[iris.feature_names]
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# TEST ACCURACY #
y_pred = lr.predict(x_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# USE SAMPLE TO PREDICT SPECIES #
print("> Enter sepal length, sepal width, petal length, petal width:")
try:
    values = list(map(float, input().split()))
    if len(values) != 4:
        print("> Please enter exactly 4 values.")
    # Create a DataFrame with proper column names
    input_df = pd.DataFrame([values], columns=iris.feature_names)
    prediction = lr.predict(input_df)
    print(f"> Prediction: {iris.target_names[prediction[0]]}")
except Exception as e:
    print("> Invalid input. Try again.")
