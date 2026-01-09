# Exploratory Data AnalysisProject

You can find the dataset:
*https://www.kaggle.com/datasets/umuttuygurr/e-commerce-customer-behavior-and-sales-analysis-tr*

## 1. Introduction
<p align="justify">
This is a e-commerce dataset, from sales across several cities in Turkiye. This dataset will be employed to make an <strong>Exploratory Data Analysis</strong>, during this project we will:
</p>
<strong>1-</strong> Import libraries </br>
<strong>2-</strong> Import the dataset </br>
<strong>3-</strong> Exploratory Data Analysis </br>
<strong>3.1-</strong> First we are going to focus in understand the dataset </br>
<strong>3.2-</strong> We will look for NaN values, duplicates and outliers in the dataset </br>
<strong>3.3-</strong> Then we will analyse the correlation among the features in the dataset </br>
<strong>3.4-</strong> Finally we are going to answer some question with financial interest like: </br>
<strong>3.4.1-</strong> Which cities generate the highest total sales in the Beauty and Fashion categories? </br>
<strong>3.4.2-</strong> How do discounts affect the quantity of items purchased? </br>
<strong>3.4.3-</strong> What are the most popular payment methods by age group? </br>
<strong>3.4.4-</strong> How does device type influence user engagement, measured by pages viewed and session duration? </br>
<strong>3.4.5-</strong> Which cities have the lowest average delivery time (most efficient delivery)? </br>


## 2. Importing Libraries 
<p align="justify">
We will be working in this dataset with 3 libraries, </strong>pandas</strong> for tabular data managment, <strong>matplotlib.pyplot</strong> and <strong>seaborn</strong> for visualization:
</p>

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Plot style that I like
plt.style.use('seaborn-v0_8-whitegrid')
```

## 3. Importing Dataset
In this step, we are going to use pandas to import the data, and use head() to explore the first five rows of the data:
```python
file_path = './data/ecommerce_customer_behavior_dataset_v2.csv'

data_sales = pd.read_csv(file_path)
data_sales.head()
```

<details>
  <summary>Click to see output of <strong>head()</strong> </summary>

<div align="center">
  <img src="Images/head.png" style="max-width: 100%; height: auto;">
</div>
<p><strong>Figure 1.</strong> First 5 rows of the dataset

</details>

## 4. Exploratory Data Analysis

### 4.1 Understanding the Dataset

We first used <strong>shape</strong> to know the size of the dataset, then we use the info() method to know the column types, and if there was any null-values in the dataset:
```python
data_sales.info()
```

<details>
  <summary>Click to see output of <strong>info()</strong> </summary>
<div align="center">
  <img src="Images/info.png" style="max-width: 100%; height: auto;">
</div>
<p><strong>Figure 2.</strong> Info method output with the dtypes of each column

</details>

<p align="justify">
We can see that we have in the dataset:
</p>

- 8 <strong>numerical</strong> columns </br>

- 8 <strong>object</strong> type columns </br>

- 1 <strong>boolean</strong> column </br>

Also we can notice that we haven't <strong>NaN values</strong> in the dataset, that can be also confirmed by looking directly into the <strong>isna().sum()</strong> output.

### 4.2 Null values, outliers and duplicates

```python
data_sales.isna().sum()
```

<details>
  <summary>Click to see output of <strong>isna().sum()</strong> </summary>
<div align="center">
  <img src="Images/isna.png" style="max-width: 100%; height: auto;">
</div>
<p><strong>Figure 3.</strong> Isna method output with 0 NaN values  

</details>

<p align="justify">
After the confirmation of the absence of <strong>null values</strong> in the dataset, we looked for <strong>outliers</strong> and <strong>duplicates</strong> in the dataset. For the first one, we built histograms with all the numerical columns in the dataset, the advantage of using this method is that allow us quickly to detect outliers in the dataset: 
```python
data_sales.hist(figsize=(12, 8), bins=30)
```
</p>

<div align="center">
  <img src="Images/hist_outliers.png" style="max-width: 100%; height: auto;">
</div>
<p><strong>Figure 4.</strong> Histograms of numerical columns to look for outliers

</details>

Then we looked for duplicates in the dataset: 
```python
data_sales.duplicated().sum()
```
<details>
  <summary>Click to see output of <strong>duplicated().sum()</strong> </summary>

<div align="center">
  <img src="Images/duplicates.png" style="max-width: 100%; height: auto;">
</div>
<p><strong>Figure 5.</strong> Duplicates output, showing the absence of duplicate entries in the dataset

</details>

- If we analyse the histograms, all the numerical features behave inside of the normal patterns, so we don't have any obvious outlier in this dataset nor duplicates. 

### 4.3 Features Correlation

<p align="justify">
There wasn't nan values present in the dataset as was confirmed by running <strong>isna().sum()</strong>. After this we explore the data, to see if all the values were in a logical range:
</p>

<div align="center">
  <img src="Images/data_exploration.png" alt="Screenshot1">
</div>
<p><strong>Figure 2.</strong> Dataset column's histograms

<p align="justify">
There wasn't any anomaly present in the features or the output classes in the dataset.
</p>

<p align="justify">
We are addressing a classification problem, the desired output labels are bad quality wine that will be represented with a 0 value and good quality wine 
that will be represented with a 1 value. Input features had different scales:
</p>

1- Fixed acidity: <strong> 0<x<16 </strong> </br>
2- Volatile acidity: <strong> 0<x<1.6 </strong> </br>
3- Citric acid concentration: <strong> 0<x<1 </strong> </br>
4- Residual sugar: <strong> 0<x<16 </strong> </br>
5- Amount of chlorides: <strong> 0<x<0.6 </strong> </br>
6- Total amount of sulfur dioxide: <strong> 0<x<75 </strong> </br>
7- Solution density: <strong> 0<x<300 </strong> </br>
8- Solution pH: <strong> 2.6<x<4.2 </strong> </br>
9- Amont of sulphates: <strong> 0.25<x<2 </strong> </br>
10- Alcohol grade: <strong> 8<x<15.2 </strong> </br>

<p align="justify">
For this reason the features  <strong>[fixed acidity, residual sugar, free sulfur dioxide, total sulfur dioxide, alcohol]</strong> were scaled using min-max normalization, after this the original columns (without scaling) were dropped, because they were not needed for the model training.
</p>

## 3. Benchamark model

<p align="justify">
A <strong>DecisionTreeClassifier</strong> was the model selected to train using the dataset. We import the model from the ML-library sklearn, after this we split the data into training and testing set, we set the size of the testing set using the hyperparameter <strong>test_size=0.2</strong>. The model was trained, accuracy metrics and a confusion matrix were used to determine the model's perfomance. The accuracy of the benchmark model was <strong>accuracy=0.725</strong> and the confusion matrix is presented in the following figure:
</p>

<div align="center">
  <img src="Images/benchmark_performance.png" alt="Screenshot2">
</div>
<p><strong>Figure 3.</strong> Benchmark Model Performance

<p align="justify">
The DecisionTreeClassifier performed well in solving this classification problem. However, its performance can be further enhanced through feature engineering and by experimenting with other models that may better fit our data. 
</p>

## 4. Feature Engineering

<p align="justify">
A correlation feature matrix was used to analyze the relationship between input features and output classes. A threshold of 0.2 was set, and all features with a correlation above this value with wine quality were selected for retraining the model. 
</p>

<div align="center">
  <img src="Images/features_selection.png" alt="Screenshot3">
</div>
<p><strong>Figure 4.</strong> Correlation Feature Matrix 

<p align="justify">
After this selection process, the input features selected were <strong>[volatile acidity, sulphates, total sulfur dioxide_scaled, alcohol_scaled]</strong>.
</p>

## 5. Model Training 

<p align="justify">
In this section three models were selected to train and compare their perfomance based in <strong>accuracy, precision, recall and f1-score metrics</strong>. The models selected were <strong> Decision Tree, Random Forest and Gradient Boosting </strong>. In the next figure are shown the confusion matrix for these three models:
</p>

<div align="center">
  <img src="Images/3_models.png" alt="Screenshot4">
</div>
<p><strong>Figure 5.</strong> Confusion Matrices for the three selected models.

<p align="justify">
In <strong>Table 1</strong>, it can be observed that the best-performing model was Random Forest.
</p>

|Metrics   |Decision Tree|Random Forest|Gradient Boosting|
|----------|-------------|-------------|-----------------|
|Accuracy  |0.737500     |0.809375     |0.734375         |
|Precision |0.787879		 |0.847059     |0.770115         |
|Recall    |0.726257		 |0.804469     |0.748603         |
|F1-score  |0.755814		 |0.825215     |0.759207         |

<p><strong>Table 1.</strong> Three models metrics comparision

## 6. Random Forest Hyperparameters Tuning 

<p align="justify">
A grid was made with several hyperparameter of the model and then used the RandomizedSearchCV function inside of sklearn library to found the best set of hyperparameters regarding to accuracy metrics. 
</p>

```python
# Number of trees in the forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=200, num=11)]
# How to compute the quality of split
criterion = ['gini', 'entropy']
# Number of features to consider at every split
max_features = ['sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [2, 4, 8, 10, 12]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 4, 8, 16, 32]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 10, 20]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'criterion': criterion,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split, 
               'min_samples_leaf': min_samples_leaf, 
               'bootstrap': bootstrap
               }

from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestClassifier(random_state=42)

# Random search
random_search = RandomizedSearchCV(rf, param_distributions=random_grid, n_iter=20, cv=5, n_jobs=-1, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

forest_opt = RandomForestClassifier(**random_search.best_params_)

```

|Metrics   |Random Forest optimized|Random Forest|
|----------|-----------------------|-------------|
|Accuracy  |0.790625               |0.809375     |
|Precision |0.829412		       |0.847059     |
|Recall    |0.787709		       |0.804469     |
|F1-score  |0.808023		       |0.825215     |

<p><strong>Table 2.</strong> Optimized and non-optimized hyperparameters Random Forest performance

## 6. Conclusions
 
 <p align="justify">
 The model's performance is already strong even before hyperparameter tuning, and further tuning does not seem to enhance it. However, there are two possible approaches to potentially improve its performance. First, increasing the <strong>n_iter</strong> hyperparameter in the <strong>RandomizedSearchCV</strong> function may help explore a wider range of parameter combinations. Second, using <strong> GridSearchCV </strong>could identify the optimal hyperparameter combination, with the inconvinient that it is computationally more expensive.
</p>