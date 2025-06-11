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








