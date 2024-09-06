
# Logistic Regression with Iris Dataset

This project demonstrates the end-to-end process of building a binary classification model using logistic regression on the Iris dataset. The model classifies between two species of Iris flowers: **versicolor** and **virginica**, after removing the `setosa` species, which is not relevant for this task. The project covers essential steps like data preparation, visualization, model training, hyperparameter tuning, and model evaluation.

## Project Overview

- **Objective**: To create a binary classification model using logistic regression to classify Iris species into either **versicolor** or **virginica**.
- **Tools Used**: Python libraries including `pandas`, `numpy`, `seaborn`, and `scikit-learn` are used for data analysis, visualization, and machine learning model creation.
- **Key Concepts**: Logistic regression, binary classification, hyperparameter tuning, and model evaluation metrics.

## Features

1. **Binary Classification**: Focuses on two Iris species (`versicolor` and `virginica`).
2. **Data Cleaning**: Prepares the data by removing one of the three classes (binary classification requirement).
3. **Data Visualization**: Uses `seaborn` to generate pair plots for visualizing feature distributions across species.
4. **Hyperparameter Tuning**: Utilizes `GridSearchCV` to optimize logistic regression parameters.
5. **Performance Evaluation**: Includes accuracy score and detailed classification metrics such as precision, recall, and F1-score.

---

## Table of Contents

1. [Libraries Used](#libraries-used)
2. [Dataset](#dataset)
3. [Workflow](#workflow)
    1. [Data Loading and Exploration](#data-loading-and-exploration)
    2. [Label Encoding](#label-encoding)
    3. [Data Visualization](#data-visualization)
    4. [Correlation Analysis](#correlation-analysis)
    5. [Data Splitting](#data-splitting)
    6. [Model Training](#model-training)
    7. [Hyperparameter Tuning](#hyperparameter-tuning)
    8. [Model Evaluation](#model-evaluation)
4. [Usage](#usage)
5. [Example Output](#example-output)
6. [File Structure](#file-structure)
7. [License](#license)

---

## Libraries Used

The following libraries are essential for the code and must be installed before running the project:

- **`pandas`**: For loading and manipulating the dataset.
- **`numpy`**: For numerical computations and data handling.
- **`seaborn`**: For loading the Iris dataset and creating visualizations such as pair plots.
- **`scikit-learn`**: Provides machine learning tools, including the logistic regression model, grid search, and evaluation metrics.

You can install these dependencies using the following command:

```bash
pip install pandas numpy seaborn scikit-learn
```

---

## Dataset

The **Iris dataset** is one of the most famous datasets in machine learning. It consists of 150 samples, each described by four features:
- **Sepal length** (cm)
- **Sepal width** (cm)
- **Petal length** (cm)
- **Petal width** (cm)

The dataset also includes a label indicating the species of Iris flower:
- `setosa`
- `versicolor`
- `virginica`

For this project, the `setosa` species is excluded to convert the problem into a binary classification task between `versicolor` and `virginica`. The dataset is loaded directly from the `seaborn` library.

---

## Workflow

### 1. Data Loading and Exploration

First, the Iris dataset is loaded using the `seaborn` library’s built-in function. A quick exploration is done to:
- Display the first few rows of the dataset.
- Identify the unique species of Iris flowers present in the dataset.
- Verify if there are any missing or null values.

```python
df = sns.load_dataset('iris')
df.head()
df['species'].unique()
df.isnull().sum()
```

### 2. Data Cleaning and Label Encoding

Since logistic regression is a binary classifier, we remove the `setosa` species from the dataset and focus only on `versicolor` and `virginica`. Then, we map the species to numerical values:
- `versicolor` is labeled as `0`.
- `virginica` is labeled as `1`.

```python
df = df[df['species'] != 'setosa']
df['species'] = df['species'].map({'versicolor': 0, 'virginica': 1})
```

### 3. Data Visualization

Using the `seaborn.pairplot()` function, we create pair plots to visualize the distribution of the features (`sepal length`, `sepal width`, `petal length`, `petal width`) across the two species. This helps us understand the data better and spot patterns or relationships between the features.

```python
sns.pairplot(df, hue='species')
```

### 4. Correlation Analysis

A correlation matrix is generated to identify relationships between the different features. This step provides insight into how the features are related to each other, which may impact model performance.

```python
df.corr()
```

### 5. Data Splitting

The dataset is split into **independent variables** (`X`) and **dependent variables** (`y`), where:
- `X`: All features (sepal length, sepal width, petal length, petal width).
- `y`: The target variable (species).

Next, we split the dataset into training and test sets, with 25% of the data reserved for testing.

```python
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
```

### 6. Model Training

We use **Logistic Regression** for binary classification. Logistic regression is a linear model that is commonly used for binary classification problems.

```python
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
```

### 7. Hyperparameter Tuning

We perform hyperparameter tuning using **GridSearchCV** to find the best parameters for the logistic regression model. The parameters that we tune include:
- **penalty**: Type of regularization (`l1`, `l2`, or `elasticnet`).
- **C**: Inverse of regularization strength.
- **max_iter**: Maximum number of iterations for model convergence.

```python
from sklearn.model_selection import GridSearchCV

parameter = {'penalty': ['l1', 'l2', 'elasticnet'], 'C': [1, 2, 3, 4, 5, 10], 'max_iter': [100, 200, 300]}

classifier_regressor = GridSearchCV(classifier, param_grid=parameter, scoring='accuracy', cv=5)
classifier_regressor.fit(x_train, y_train)
```

### 8. Model Evaluation

We evaluate the model by making predictions on the test set. The performance is measured using accuracy score, precision, recall, and F1-score, which are provided by the classification report.

```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = classifier_regressor.predict(x_test)
score = accuracy_score(y_pred, y_test)
print(f"Test Accuracy: {score}")
print(classification_report(y_pred, y_test))
```

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/iris-logistic-regression.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python logistic_regression.py
   ```

---

## Example Output

After running the script, you will see output similar to this:

```
Best Parameters: {'C': 1, 'max_iter': 100, 'penalty': 'l2'}
Best Cross-Validation Score: 0.96
Test Accuracy: 0.94

Classification Report:
              precision    recall  f1-score   support

           0       0.93      0.93      0.93        15
           1       0.95      0.95      0.95        19

    accuracy                           0.94        34
   macro avg       0.94      0.94      0.94        34
weighted avg       0.94      0.94      0.94        34
```

---

## File Structure

```
├── README.md               # Project overview and instructions
├── logistic_regression.py   # Python script with the entire logistic regression pipeline
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
