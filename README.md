# Exploratory Data Analysis Project

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
<strong>4-</strong> Take home notes </br>


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
  <img src="Images/head.png" style="max-height: 600px; width: auto;">
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
We can see that we have in the dataset: </br>


- 8 <strong>numerical</strong> columns </br>

- 8 <strong>object</strong> type columns </br>

- 1 <strong>boolean</strong> column </br>

Also we can notice that we haven't <strong>NaN values</strong> in the dataset, that can be also confirmed by looking directly into the <strong>isna().sum()</strong> output.

</p>

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
</p>

```python
data_sales.hist(figsize=(12, 8), bins=30)

```

<div align="center">
  <img src="Images/hist_outliers.png" style="max-width: 100%; height: auto;">
</div>
<p><strong>Figure 4.</strong> Histograms of numerical columns to look for outliers

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

<p align="justify">
- If we analyse the histograms, all the numerical features behave inside of the normal patterns, so we don't have any obvious outlier in this dataset nor duplicates. 
</p>

### 4.3 Features Correlation

<p align="justify">
Main goal here is to analyse correlation among features using several kind of graphs, like <strong>heatmaps</strong>, <strong>barcharts</strong> and finally a <strong>linechart</strong> for sales
evolution over time:
</p>

```python
num_variables_corr = ['Unit_Price', 'Quantity', 'Discount_Amount', 'Customer_Rating', 'Year', 'Month']
corr_matrix = data_sales[num_variables_corr].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, 
            annot=True, 
            fmt=".2f")

plt.title('Correlation Matrix of Some Numerical Variables', fontsize=16)
plt.show()
```

<div align="center">
  <img src="Images/feature_corr.png" alt="Screenshot1">
</div>
<p><strong>Figure 6.</strong> Some features Pearson's correlation heatmap

<p align="justify">
Since we are using a Pearson correlation matrix, the analysis captures linear relationships between numerical variables only. Based on the results, the following insights can be drawn: </br>

- There is a strong positive correlation between Unit Price and Discount Amount, as well as between Discount Amount and Quantity sold, suggesting that higher-priced items tend to receive larger discounts and that discounts are associated with increased sales volume. </br>

- Although the heatmap shows no correlation between Unit Price and Quantity, this does not imply the absence of a relationship. Rather, it indicates that there is no strong linear relationship between these variables. </br>

- The correlation matrix also suggests no significant linear relationship between Discount Amount and Month, indicating that discounts do not follow a consistent seasonal pattern across the year. </br>

</p>

```python
product_total_sales = data_sales.groupby('Product_Category')['Total_Amount'].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=product_total_sales.index,
    y=product_total_sales.values,
    palette='viridis',
    hue=product_total_sales.index,
    legend=False
)

plt.xlabel('Product Category', fontsize=14)
plt.ylabel('Total Sales Amount', fontsize=14)
plt.title('Total Sales by Product Category', fontsize=16)
plt.show()
```

<div align="center">
  <img src="Images/sales_per_category.png" alt="Screenshot1">
</div>
<p><strong>Figure 7.</strong> Total sales distribution per category

<p align="justify">
- The graph shows that customers spend the most on electronics, while books account for the lowest total sales.
</p>

```python
sales_per_month = data_sales.groupby(['Month', 'Year'])['Total_Amount'].sum().unstack()

sales_per_month = (
    data_sales
    .query('Year == 2023')
    .groupby('Month')['Total_Amount']
    .sum()
)

plt.figure(figsize=(10, 6))
sns.lineplot(
    x=sales_per_month.index,
    y=sales_per_month.values,
    marker='o', 
    color='teal',
)

month_dic = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
    5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
    9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}

plt.xlabel('Month', fontsize=14, labelpad=10)
plt.ylabel('Total Sales Amount', fontsize=14, labelpad=10)
plt.title('Total Sales per Month in 2023', fontsize=16)
plt.xticks(sales_per_month.index, labels=[month_dic[m] for m in sales_per_month.index])
plt.show()
```

<div align="center">
  <img src="Images/sales_time_evo.png" alt="Screenshot1">
</div>
<p><strong>Figure 8.</strong> Total sales time evolution in the year 2023

<p align="justify">

Insights drawn from the graph: </br>

- Sales reach their highest level in December, which may be driven by holiday-related spending, such as Christmas promotions and gift purchases. </br>

- A sharp decline in January follows the December peak, potentially reflecting post-holiday budget adjustments and reduced consumer spending. </br>

- After the early-year slowdown, sales gradually increase toward the beginning of the summer period, possibly influenced by seasonal demand and vacation-related purchases.</br>

</p>

### 4.4 Responding Initial Questions

#### 4.4.1 Which cities generate the highest total sales in the Beauty and Fashion categories? 

<p align="justify">

For answering this question first, we are going to filter the data using a query(), selection just the Fashion and Beauty categories and the will use  groupby().sum(), to calculate the total amount of sales in both categories:

</p>

```python
city_beauty_fashion = (data_sales.query('Product_Category == "Beauty" or Product_Category == "Fashion"')
                       .groupby('City')['Total_Amount']
                       .sum()
                       .sort_values(ascending=False)
                       )

plt.figure(figsize=(10, 8))

sns.barplot(
    x=city_beauty_fashion.index,
    y=city_beauty_fashion.values,
    palette='magma',
    hue=city_beauty_fashion.index,
    legend=False
)

plt.xlabel('City', fontsize=14, labelpad=10)
plt.ylabel('Total Sales Amount', fontsize=14, labelpad=10)
plt.title('Total Sales in Beauty and Fashion by City', fontsize=16)
plt.show()
```

<div align="center">
  <img src="Images/question1.png" alt="Screenshot1">
</div>
<p><strong>Figure 9.</strong> Top cities in Beauty and Fashion purchases

<p align="justify">

- Istanbul records the highest sales in beauty and fashion, likely influenced by its population size and market scale. </br>

- Although Adana has a smaller population than Antalya and Konya, it exhibits comparatively higher fashion and beauty spending, indicating potential differences in consumer preferences or income distribution. </br>

</p>

#### 4.4.2 How do discounts affect the quantity of items purchased?

<p align="justify">

In this case, we are going to use a scatterplot to correlate the discount with the unit price, classifing the point using the amount of items sold: 

</p>

<div align="center">
  <img src="Images/question2.png" alt="Screenshot1">
</div>
<p><strong>Figure 10.</strong> Influence of discount in amount of items purchased

<p align="justify">

- In the graph is shown that for higher discounts, people are buying more instances of the same product

</p>

#### 4.4.3 What are the most popular payment methods by age group?

<p align="justify">

To answer this question, we are going first of all to create a new column that is the Age_Group, and then we used groupby() to count the amount of customers that have payed with certain payment method:

</p>

```python
# Creating Age_Group column
data_sales['Age_Group'] = data_sales['Age'].apply(lambda x: 'Young Adults' if x < 30 else ('Adults' if x < 60 else 'Senior'))

payment_methods_age = (data_sales.groupby('Age_Group')['Payment_Method']
                       .value_counts(normalize=True)
                       .rename('Proportion')
                       .reset_index())
payment_methods_age


plt.figure(figsize=(10, 8))

sns.barplot(
    data=payment_methods_age,
    x='Payment_Method',
    y='Proportion',
    hue='Age_Group',
    palette='mako'
)

plt.xlabel('Payment Method', fontsize=14, labelpad=10)
plt.ylabel('Proportion', fontsize=14, labelpad=10)
plt.title('Proportion of Payment Methods by Age Group', fontsize=16)
plt.legend(title='Age Group', title_fontsize=12, frameon=True)
plt.show()

```

<div align="center">
  <img src="Images/question3.png" alt="Screenshot1">
</div>
<p><strong>Figure 11.</strong> Influence of age in payment method selected

<p align="justify">

- Credit cards are the most commonly used payment method across all age groups, while only about 5% of customers in each age category pay in cash

</p>

#### 4.4.4 How does device type influence user engagement, measured by pages viewed and session duration?

In this case, we used two boxchart subplots, in the first one we analyse the influence of the device type with the amount of websited visited, and in the second one also the influence of the divice type but over the time that the customers spent in their sessions:

<div align="center">
  <img src="Images/question4.png" alt="Screenshot1">
</div>
<p><strong>Figure 12.</strong> Influence of device type pages viewed and session duration

<p align="justify">

- Customers predominantly use mobile devices for online purchases in this store, visiting on average around 9 pages before completing a purchase, with a maximum of 11 pages viewed. </br>

- However, device type does not appear to strongly influence the amount of time users spend online before making a purchase, with a median session duration of 15 minutes across all three device types. </br>

</p>

#### 4.4.5 Which cities have the lowest average delivery time (most efficient delivery)?

<p align="justify">

For this question, we first grouped by City, the aggrupate cities by their counts and the total delivery time in days (sum), afterwards we created a new column Delay_per_order, with the average delay per order: 

</p>

```python

# Calculating average delivery delay per order by city
rank_delay_df = (
    data_sales
    .groupby('City')
    .agg(
        Number_of_Orders=('City', 'size'),
        Total_delivery_time=('Delivery_Time_Days', 'sum')
    )
    .assign(Delay_per_order=lambda x: x['Total_delivery_time'] / x['Number_of_Orders'])
    .sort_values('Delay_per_order', ascending=False)
)

plt.figure(figsize=(10, 6))
sns.barplot(data = rank_delay_df,
            x=rank_delay_df.index, 
            y='Delay_per_order', 
            palette='viridis',
            hue = rank_delay_df.index,
            legend=False)
plt.xlabel('City', fontsize=14, labelpad=10)
plt.ylim([0,8])
plt.ylabel('Avg Delay per Order (Days)', fontsize=14, labelpad=10)
plt.title('Average Delivery Delay per Order', fontsize=16)
plt.show()

```

<div align="center">
  <img src="Images/question5.png" alt="Screenshot1">
</div>
<p><strong>Figure 13.</strong> Average delay in days per order

<p align="justify">

- Konya and Kayseri are among the smallest cities in the dataset, however they exhibit the highest average delivery delays. </br>
- Overall, the average delivery delay is approximately six days across all cities. </br>

</p>

#### 5. Conclusions