# Databricks notebook source
# MAGIC %md
# MAGIC # RADARSAT-1 Ice Coverage Analysis
# MAGIC 
# MAGIC This script uses data from the satellite and known measurements of ice on various lakes to attempt to make an AI model that predicts ice coverage, given a photo.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install GDAL
# MAGIC 
# MAGIC We install GDAL using wheels available online. This is done using Jupyter Notebook's CLI functionality.

# COMMAND ----------

# MAGIC %pip install https://manthey.github.io/large_image_wheels/GDAL-3.5.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl#sha256=9387d6f4a71a132a7c5a13426a2491e9aded5e0974cadb43b9d579fac92541f8

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import Packages and Setup

# COMMAND ----------

import cv2
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import fnmatch
import random
import pickle
from datetime import datetime
from osgeo import gdal, gdal_array
from pathlib import Path

import sklearn as sk
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Code

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read the CSV Files

# COMMAND ----------

df_r1 = pd.read_csv('./r1_data_with_aws.csv') # Reads the R1 metadata file
df = pd.read_csv("./lakeice_measurements.csv") # Reads the data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Adding Needed Information
# MAGIC 
# MAGIC We then add columns that identify the coordinates and date of the measurement given.

# COMMAND ----------

def get_coord_1(row):
    row = str(row)
    x = row.split('(')[1].split(' ')[0]
    return x

def get_coord_2(row):
    row = str(row)
    y = row.split('(')[1].split(' ')[1].split(')')[0]
    return y

def get_year(row):
    if type(row) == str:
        return row.split('-')[0]
    return 0

def get_month(row):
    if type(row) == str:
        return row.split('-')[1]
    return 0

# COMMAND ----------

df_r1['long'] = [get_coord_1(row) for row in df_r1['scene-centre']] # Changes the coordinates to separate columnshttps://adb-588851212245547.7.azuredatabricks.net/?o=588851212245547#
df_r1['lat'] = [get_coord_2(row) for row in df_r1['scene-centre']] # Changes the coordinates to separate columns
df_r1['month'] = [get_month(row) for row in df_r1['start-date']] # Changes the date to separate columns
df_r1['year'] = [get_year(row) for row in df_r1['start-date']] # Changes the date to separate columns
df_r1 = df_r1.astype({'long': 'float64', 'lat': 'float64', 'month': 'int64', 'year': 'int64'})

df['has_coverage'] = [has_imagery(row, df_r1) for row in df.iterrows()]

output_df = df[df['has_coverage'].notnull()]
output_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Eliminating Unwanted Data
# MAGIC 
# MAGIC We only want to measure the instances where 0% or 100% of the lake is covered in ice. We eliminate those with other measures and also ensure only unique images are used.

# COMMAND ----------

output_zeros = output_df[output_df['ICE_COVER'] == 0].copy()
output_hundreds = output_df[output_df['ICE_COVER'] == 10].copy()

unique_0_images = output_zeros['has_coverage'].unique()
unique_100_images = output_hundreds['has_coverage'].unique()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create and Display Histograms
# MAGIC 
# MAGIC We then convert our images to arrays, which will allow us to display them through histograms.

# COMMAND ----------

def image_to_array(location_string):
    img_ds = gdal.Open(location_string, gdal.GA_ReadOnly)
    band = img_ds.GetRasterBand(1)
    img = band.ReadAsArray()
    return img

def image_to_linear_array(location_string, increment):
    matrix = image_to_array(location_string)
    if (sum(sum(matrix))) > 10000000:
        temp = np.hstack(matrix)//255    
        return np.histogram(temp, bins=np.arange(start=0, stop=256, step=increment))
    return np.histogram(np.hstack(matrix), bins=np.arange(start=0, stop=256, step=increment))

# COMMAND ----------

INCREMENT = 8
bins = 0
arrays_100 = []
arrays_0 = []

for file in unique_100_images:
    linarr, bins = image_to_linear_array(file, INCREMENT)
    arrays_100.append(linarr)
for file in unique_0_images:
    linarr, bins = image_to_linear_array(file, INCREMENT)
    arrays_0.append(linarr)

total_100_linarr = [0]*(len(bins)-1)
for array in arrays_100:
    total_100_linarr = np.add(total_100_linarr, array)
total_0_linarr = [0]*(len(bins)-1)
# Random sampling 17 files as there are only 17 valid 100% ICE_COVER files, so remaining consistent with 0% ICE_COVER files.
for array in random.sample(arrays_0, 17):
    total_0_linarr = np.add(total_0_linarr, array)
    
fig, ax1 = plt.subplots(1, 1)
fig.set_figheight(8)
fig.set_figwidth(15)
width = 4
rects1 = ax1.bar(height=total_100_linarr, x=bins[:-1]-width/2, width=width, label="100%")
ax1.set_title('100%')
rects2 = ax1.bar(height=total_0_linarr, x=bins[:-1]+width/2, width=width, label="0%")
ax1.legend()

# COMMAND ----------

df_100 = pd.DataFrame(arrays_100)
df_0 = pd.DataFrame(random.sample(arrays_0, 17))

fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2)
fig.set_figheight(8)
fig.set_figwidth(15)
ax100 = df_100.plot(kind="box", ax=ax1)
df_0.plot(kind="box", ylim = ax1.get_ylim(), ax=ax2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### AI Model
# MAGIC 
# MAGIC We train an SVM and a Gaussian Naive Bayes model and then test it on a test dataset. Scores are output below them.

# COMMAND ----------

df_0['ICE'] = 0
df_100['ICE'] = 1
df_all = pd.concat([df_100, df_0])

y = df_all.iloc[:,31]
X = df_all.iloc[:,:31]

SVM = svm.SVC()
SVM.fit(X, y)

filename = 'finalized_model.sav'
pickle.dump(SVM, open(filename, 'wb'))

print('SVM Score:', round(SVM.score(X,y), 4))

# COMMAND ----------

clf = GaussianNB()
clf.fit(X, y)
print('GNB Score:', round(clf.score(X,y), 4))
