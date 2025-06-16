# Machine Learning with the Iris Dataset

## Problem Statement

The Iris Dataset is the most popular beginner project in machine learning. It was collected in 1936 and contains 50 samples of petal length and width, and sepal length and width of Iris flowers. The dataset is small, clean, and easy to work with. The goal of the project is to use machine learning (ML) to create a predictive model, based on sample measurements, that determines which species the sample belongs to. The Iris Dataset is a sandbox for testing ML models and statistical methods. It also proves useful for learning visualization.

![Sepals](https://github.com/user-attachments/assets/08bad192-0a7a-40b2-965b-ce2cf622ab71)

---

## Exploratory Data Analysis (EDA)

Even for such a simple dataset, initial exploration is important for understanding what data is being stored.

~~~
print(df.info())
~~~

| # | Column            | Non-Null Count | Dtype    |
|---|-------------------|----------------|----------|
| 0 | sepal length (cm) | 150 non-null   | float64  |
| 1 | sepal width (cm)  | 150 non-null   | float64  |
| 2 | petal length (cm) | 150 non-null   | float64  |
| 3 | petal width (cm)  | 150 non-null   | float64  |
| 4 | target            | 150 non-null   | int32    |
| 5 | target_name       | 150 non-null   | category |

---

~~~
print(df.describe())
~~~

|         | sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) |
|---------|-------------------|------------------|-------------------|------------------|
| `count` | 150.00            | 150.00           | 150.00            | 150.00           |
| `mean`  | 5.84              | 3.06             | 3.76              | 1.20             |
| `std`   | 0.82              | 0.44             | 1.77              | 0.76             |
| `min`   | 4.30              | 2.00             | 1.00              | 0.10             |
| `25%`   | 5.10              | 2.80             | 1.60              | 0.30             |
| `50%`   | 5.80              | 3.00             | 4.35              | 1.30             |
| `75%`   | 6.40              | 3.30             | 5.1               | 1.80             |
| `max`   | 7.9               | 4.40             | 6.9               | 2.5              |

---

~~~
print(df['target_name'].value_counts())
~~~

| target_name |    |
|-------------|----|
| setosa      | 50 |
| versicolor  | 50 |
| virginica   | 50 |

---

## Visualization 

```
selected_features = ['petal length (cm)', 'petal width (cm)', 'sepal length (cm)', 'sepal width (cm)', 'target_name']
sns.pairplot(df[selected_features], hue='target_name', palette='Set1')
plt.suptitle('Feature Relationships in the Iris Dataset')
plt.show()
```

![iris-dataset-pairplot](https://github.com/user-attachments/assets/288da9ee-aa2d-406d-bb5e-710599287842)

---

## Building the ML Model

Sklearn is used to split the data and train the model for predicting the Iris species. We split the data into an x and a y axis, where the x-axis is the features of the Iris, and the y-axis is the target species. We use 80% of the data to train the model and 20% to test the accuracy. We are left with four variables: x_train, x_test, y_train, y_test.

```
x = df[iris.feature_names]
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

Next, a logistic regression algorithm is used to train the model and fit the training data.

```
lr = LogisticRegression()
lr.fit(x_train, y_train)
```

We can check the accuracy of the model using a confusion chart. The diagonal of the matrix represents correct predictions, and the off diagonals are confusions.

```
y_pred = lr.predict(x_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

```math
\begin{bmatrix} 10 & 0 & 0 \\ 0 & 9 & 0  \\ 0 & 0 & 11 \end{bmatrix}
```

The confusion matrix shows that our model predicted all values from our test split correctly. This is to be expected as the Iris dataset is small, simple, and clean. We can also use a classification report and accuracy score to get a better quantatative measure of the models accuracy.

```
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
```

This comes out to be 1.00, meaning 100% accuracy.

---

## Conclusion

The Iris Dataset is an excellent introductory tool for exploring machine learning concepts, as demonstrated in this project. By performing exploratory data analysis (EDA), we gained an understanding of the dataset's structure and relationships between its features. The visualization step revealed clear separability between the three Iris species based on their petal and sepal measurements, setting a strong foundation for building a predictive model.

Using logistic regression, we trained a simple yet effective classifier to predict Iris species with 100% accuracy on the test set. This result is expected given the simplicity and cleanliness of the dataset, as well as the linear separability of its classes. The confusion matrix confirms that the model made no misclassifications, while additional metrics, such as the classification report and accuracy score, reinforce the reliability of the model.

This project highlights the importance of EDA, visualization, and robust evaluation in the machine learning pipeline. Although the Iris Dataset is not representative of real-world complexity, it serves as a valuable sandbox for understanding key concepts and methodologies. Future directions could involve applying more complex models, exploring feature engineering, or testing on more challenging datasets to simulate practical scenarios.







