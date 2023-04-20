# Databricks notebook source
# MAGIC %md
# MAGIC # Classification Using Measurement Data
# MAGIC 
# MAGIC This notebook contains a basic classification that analyzes the lake ice measurements provided by Environment and Climate Change Canada and the Canadian Ice Service program.
# MAGIC 
# MAGIC This was done as part of exploratory work into using this dataset with basic machine learning.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# Import the required packages for data analysis and machine learning
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC We start by reading the lake ice measurements from the file. We create a "HAS_ICE" column that determines whether a measurement has ice or not, which can then be used for training and testing.

# COMMAND ----------

X = pd.read_csv("./lakeice_measurements.csv")
X['DATE'] = pd.to_datetime(X['DATE'], errors='coerce')
X['YEAR'] = X['DATE'].dt.year
X['MONTH'] = X['DATE'].dt.month
X['HAS_ICE'] = X['ICE_COVER'] > 0.5
y = X.pop("HAS_ICE").values
X.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC We then create a feature set, using the latitude, longitude, year and month of the lake measurements as features. We then randomly split the data into a training and test set.

# COMMAND ----------

featureSet = ['LAT', 'LONG', 'YEAR', 'MONTH']
X = X[featureSet].copy()

# split the large dataset into train and test
print("Splitting the dataset...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=2)
print("Done!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Naive Bayes
# MAGIC 
# MAGIC We can then apply a Naive Bayes algorithm to see if it can be trained on the data well.

# COMMAND ----------

# Helper to calculate accuracy
def accuracy(actualTags, predictions):
    totalFound = 0
    for i in range(len(actualTags)):
        if (actualTags[i] == predictions[i]):
            totalFound += 1
    return totalFound / len(predictions)

# COMMAND ----------

print("Training the NB classifier...")
clf_nb = MultinomialNB().fit(X_train, y_train)
print("Done!")

# COMMAND ----------

training_predictions = clf_nb.predict(X_train)
print(training_predictions[0:10])
print(accuracy(y_train, training_predictions))

# COMMAND ----------

testing_predictions = clf_nb.predict(X_val)
print(testing_predictions[0:10])
print(accuracy(y_val, testing_predictions))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Logistic Regression
# MAGIC 
# MAGIC Same as the Naive Bayes algorithm but using a Logistic Regression algorithm instead.

# COMMAND ----------

print("Training the LR classifier...")
clf_lr = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=1).fit(X_train, y_train)
print("Done!")

# COMMAND ----------

training_predictions_lr = clf_lr.predict(X_train)
print(training_predictions_lr[0:10])
print(accuracy(y_train, training_predictions_lr))

# COMMAND ----------

testing_predictions_lr = clf_lr.predict(X_val)
print(testing_predictions_lr[0:10])
print(accuracy(y_val, testing_predictions_lr))

# COMMAND ----------

data = [[45, 75, 2013, 1], [45, 75, 2013, 2], [45, 75, 2013, 3], [45, 75, 2013, 4], [45, 75, 2013, 5], [45, 75, 2013, 6], [45, 75, 2013, 7], [45, 75, 2013, 8], [45, 75, 2013, 9], [45, 75, 2013, 10], [45, 75, 2013, 11], [45, 75, 2013, 12]]
df_test = pd.DataFrame(data, columns = ['LAT', 'LONG', 'YEAR', 'MONTH'])
clf_nb.predict(df_test)
