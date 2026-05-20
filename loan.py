#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[6]:


sns.set_theme(style="whitegrid")


# In[7]:


loan = pd.read_csv("loan_data.csv")
print(loan.head(10))


# In[8]:


print("Data dimensions:", loan.shape)


# In[9]:


print("\n--- Data Info ---")
loan.info()


# In[10]:


print("\n--- Summary Statistics ---")
print(loan.describe(include='all'))


# In[11]:


# summary of the numeric data
print("\n--- Numeric Summary ---")
print(loan.select_dtypes(include=[np.number]).describe())

# summary of factor/categorical variables
print("\n--- Categorical Summary ---")
print(loan.select_dtypes(include=['object', 'category']).describe())

# looking for the variables with NAs
print("\n--- Missing values per column ---")
print(loan.isnull().sum())

# total number of NAs in the entire dataframe
print("\nTotal missing values:", loan.isnull().sum().sum())


# In[12]:


# Histogram for person_age filled by person_education
plt.figure(figsize=(10, 6))
sns.histplot(data=loan, x='person_age', hue='person_education', element='step', stat='count', common_norm=False)
plt.title('Histogram of Age by Education')
plt.show()

# Histograms for all numeric columns
numeric_cols = loan.select_dtypes(include=[np.number]).columns
loan[numeric_cols].hist(figsize=(12, 10), bins=20)
plt.suptitle('Histograms for all Numeric Data')
plt.show()

# Barchart: person_home_ownership vs person_income filled by person_gender
plt.figure(figsize=(10, 6))
sns.barplot(data=loan, x='person_home_ownership', y='person_income', hue='person_gender', errorbar=None)
plt.title('Income by Home Ownership and Gender')
plt.show()

# Scatter plot with smooth trend line: person_age vs person_income colored by person_home_ownership
sns.lmplot(data=loan, x='person_age', y='person_income', hue='person_home_ownership', height=6, aspect=1.5, scatter_kws={'alpha':0.5}, line_kws={'linewidth': 2})
plt.title('Age vs Income by Home Ownership')
plt.show()

# Smooth trend line: person_age vs person_income colored by person_gender
plt.figure(figsize=(10, 6))
sns.lineplot(data=loan, x='person_age', y='person_income', hue='person_gender', errorbar=None)
plt.title('Age vs Income Trend line by Gender')
plt.show()

# Frequency polygon (approximated with a line-based histogram)
plt.figure(figsize=(10, 6))
sns.histplot(data=loan, x='person_age', hue='person_education', element='poly', fill=False, binwidth=10)
plt.title('Frequency Polygon of Age by Education')
plt.show()


# In[13]:


### Data cleaning
# If age > 100, income > 7,200,764, or experience > 124, replace with NaN
loan.loc[loan['person_age'] > 100, 'person_age'] = np.nan
loan.loc[loan['person_income'] > 7200764, 'person_income'] = np.nan
loan.loc[loan['person_emp_exp'] > 124, 'person_emp_exp'] = np.nan

print(loan.describe(include='all'))

# Group by education and find mean income
mean_income_by_edu = loan.groupby('person_education')['person_income'].mean().reset_index()
print(mean_income_by_edu.sort_values(by='person_income', ascending=False))

# Bar graph showing count of person_education
plt.figure(figsize=(8, 5))
sns.countplot(data=loan, x='person_education', hue='person_education', legend=False)
plt.title('Count of Person Education')
plt.show()

# Bar chart for loan intent sorted by counts
plt.figure(figsize=(10, 5))
order = loan['loan_intent'].value_counts().index
sns.countplot(data=loan, x='loan_intent', order=order, hue='loan_intent', legend=False)
plt.title('Count for Loan Intent')
plt.xticks(rotation=45)
plt.show()

# Bar chart for loan intent crossed with person_gender
plt.figure(figsize=(12, 6))
sns.countplot(data=loan, x='loan_intent', hue='person_gender', order=order)
plt.title('Loan Intent by Gender')
plt.xticks(rotation=45)
plt.show()

# Cross table of loan_intent vs person_gender
print("\n--- Contingency Table: Intent vs Gender ---")
print(pd.crosstab(loan['loan_intent'], loan['person_gender']))

# Scatter plot: person_income vs person_age colored by education
plt.figure(figsize=(10, 6))
sns.scatterplot(data=loan, x='person_income', y='person_age', hue='person_education', alpha=0.2)
plt.title('Income vs Age colored by Education')
# Avoiding plotting outliers implicitly by using plot limits if required, matching 'outlier.shape = NA'
plt.show()

# Log-scaled boxplot with tomato jitter points
plt.figure(figsize=(10, 6))
ax = sns.boxplot(data=loan, x='person_education', y='person_income', showfliers=False)
sns.stripplot(data=loan, x='person_education', y='person_income', color='tomato', alpha=0.4, jitter=0.2, ax=ax)
ax.set_yscale('log')
plt.title('Log Boxplot of Income by Education with Jitter')
plt.show()

# Filter, select, sort, and head
filtered_loan = loan[loan['person_age'].notnull()][['person_age', 'person_income', 'person_education']]
print(filtered_loan.sort_values(by='person_income', ascending=False).head())


# In[14]:


# Create a copy of the dataframe to encode features cleanly
encoded_loan = loan.copy()

# Identify categorical columns (strings/objects)
categorical_cols = encoded_loan.select_dtypes(include=['object']).columns

# One-hot encode features, saving the target 'loan_status' as numeric if it's string
# Assuming loan_status is a binary flag indicator (0 or 1) or text (e.g. 'Y'/'N')
if encoded_loan['loan_status'].dtype == 'object':
    encoded_loan['loan_status'] = pd.factorize(encoded_loan['loan_status'])[0]

# Convert features into numeric dummy features
X = pd.get_dummies(encoded_loan.drop(columns=['loan_status']), drop_first=True)
y = encoded_loan['loan_status']


# In[15]:


# Check class target proportions overall
print("Overall Target Proportions:\n", y.value_counts(normalize=True).round(2))

# Train-Test Split (75% Train, 25% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)

print("Train Target Proportions:\n", y_train.value_counts(normalize=True).round(2))
print("Test Target Proportions:\n", y_test.value_counts(normalize=True).round(2))

### Training the decision tree model
# Using standard entropy classification matching standard recursive partitioning frameworks
dt_mod = DecisionTreeClassifier(criterion='entropy', random_state=1234, max_depth=4) 
dt_mod.fit(X_train, y_train)

### Evaluating the model
plt.figure(figsize=(20, 10))
plot_tree(dt_mod, feature_names=X.columns, class_names=True, filled=True, fontsize=10)
plt.title("Decision Tree Structure")
plt.show()

### Prediction on test data
dt_pred = dt_mod.predict(X_test)

print("\n--- Decision Tree Confusion Matrix ---")
dt_cm = confusion_matrix(y_test, dt_pred)
print(dt_cm)

dt_acc = accuracy_score(y_test, dt_pred)
print(f"Decision Tree Accuracy: {dt_acc:.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, dt_pred))


# In[16]:


### Build the model for randomForest algorithm
# Drop rows with any missing data (equivalent to na.omit(loan))
clean_loan = loan.dropna()

# Encode features for the fresh rows
if clean_loan['loan_status'].dtype == 'object':
    clean_loan['loan_status'] = pd.factorize(clean_loan['loan_status'])[0]

X_rf = pd.get_dummies(clean_loan.drop(columns=['loan_status']), drop_first=True)
y_rf = clean_loan['loan_status']

### Splitting the data
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.25, random_state=1234)

### Building the Random Forest model
# R equivalents: mtry = 4 (max_features=4), ntree = 2001 (n_estimators=2001)
rf_mod = RandomForestClassifier(n_estimators=2001, max_features=4, random_state=1234, n_jobs=-1)
rf_mod.fit(X_train_rf, y_train_rf)

### Prediction on train data
rf_train_pred = rf_mod.predict(X_train_rf)
print("\n--- RF Train Confusion Matrix ---")
print(confusion_matrix(y_train_rf, rf_train_pred))
print("Train Accuracy:", accuracy_score(y_train_rf, rf_train_pred))

### Prediction on test data
rf_test_pred = rf_mod.predict(X_test_rf)
print("\n--- RF Test Confusion Matrix ---")
print(confusion_matrix(y_test_rf, rf_test_pred))
rf_test_acc = accuracy_score(y_test_rf, rf_test_pred)
print(f"Random Forest Test Accuracy: {rf_test_acc:.4f}")

### Important variables
importances = rf_mod.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 8))
plt.title('Feature Importances (Variable Importance Plot)')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [X_rf.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

print("\n--- Final Analysis Conclusion ---")
print(f"Decision Tree Accuracy: {dt_acc * 100:.2f}%")
print(f"Random Forest Accuracy: {rf_test_acc * 100:.2f}%")
print("In conclusion, the random forest algorithm is best suited for this data since it provides higher test classification accuracy.")

