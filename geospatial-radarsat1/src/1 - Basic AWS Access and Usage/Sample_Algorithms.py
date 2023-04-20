# Databricks notebook source
# MAGIC %md
# MAGIC # Sample Algorithms
# MAGIC 
# MAGIC These are a set of algorithms that show some of the things you can do with the mounted S3 bucket of RADARSAT-1 imagery

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Map of the Data
# MAGIC 
# MAGIC This creates a visual representation of where imagery was taken around the world

# COMMAND ----------

# MAGIC %run "./Get_Metadata"

# COMMAND ----------

!pip install opencv-contrib-python

# COMMAND ----------

import plotly.express as px

# COMMAND ----------

def create_a_map(start_year, start_month, end_year=-1, end_month=-1):
    """Creates a map of data from a given year and month.
    :param start_year: The year to start the map from.
    :param start_month: The month to start the map from.
    :param end_year: The year to end the map at.
    :param end_month: The month to end the map at.
    """
    lat = []
    long = []

    # If no end year and month are given, get data from the current month.
    if end_year == -1 or end_month == -1:
        end_year = start_year
        end_month = start_month

    # If the start year and month are the same as the end year and month, we only want to get data from one month.
    if (start_year == end_year and start_month == end_month):
        df = get_data_from_month_and_year(start_year, start_month)
    # Otherwise, we want to get data from a date range.
    else:
        df = get_data_from_date_range(start_year, start_month, end_year, end_month)
        
    df = df.toPandas()

    # Get the latitude and longitude of the data.
    for x in df["scene-centre"]:
        temp = x.split("(")[1].split(" ")
        temp[1] = temp[1].replace(")", "")

        # Append the latitude and longitude to the list.
        long.append(float(temp[0]))
        lat.append(float(temp[1]))

    # Add the latitude and longitude to the dataframe.
    df['Latitude'] = lat
    df['Longitude'] = long

    # Create a map of the data.
    fig = px.scatter_geo(df, lat="Latitude", lon="Longitude", opacity=0.5)
    fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Demo
# MAGIC 
# MAGIC This demonstrates the above function by creating a map of all imagery taken in the first half of 2006.

# COMMAND ----------

create_a_map(2006, 1, 2006, 12)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Detect Land/Water
# MAGIC 
# MAGIC This algorithm roughly attempts to find the water/land borders in an image. It does this by analyzing the difference in brightness of water compared to land.

# COMMAND ----------

import cv2
import numpy as np
import matplotlib.pyplot as plt

beta = 0

# COMMAND ----------

def basicLinearTransform(img_original, alpha=1.0):
    '''Applies a basic linear transform to the image.
    :param img_original: The image to be transformed.
    :param alpha: The alpha value to be used in the transform.
    '''
    res = cv2.convertScaleAbs(img_original, alpha=alpha, beta=beta)
    return res

# COMMAND ----------

def gammaCorrection(img_original, gamma=1.0):
    '''Applies a gamma correction to the image.
    :param img_original: The image to be transformed.
    :param gamma: The gamma value to be used in the transform.
    '''
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(img_original, lookUpTable)
    return res

# COMMAND ----------

def borders(img_url):
    '''Attempts to find the borders of the image.
    :param img_url: The url of the image to be transformed.
    '''
    img = cv2.imread(img_url)
    img = cv2.resize(img, (500, 500), interpolation = cv2.INTER_AREA)

    # Convert to grayscale.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Applies a gaussian blur to the image.
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Applies a basic linear transform to the image.
    img_new = basicLinearTransform(img_blur, 4.0)

    # Applies a bilateral filter to the image.
    img_filtered = cv2.bilateralFilter(img_new, 7, 50, 50)

    # Attempts to find the edges of the image.
    thresh = 60
    edges = cv2.Canny(image=img_filtered, threshold1=thresh, threshold2=thresh*3)

    # Outputs the images.
    output = np.hstack((img_gray, img_blur, img_new, img_filtered, edges))
    plt.imshow(output)

    cv2.waitKey(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Demo
# MAGIC 
# MAGIC This demonstrates using an image taken off the BC coast.

# COMMAND ----------

borders('/dbfs/mnt/test_radarsat_mount/2007/8/RS1_M0620349_SCNA_20070825_020117_HH_SCN.tif')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chart Amount of Imagery by Date

# COMMAND ----------

def chart_imagery_by_date(start_year=1996, start_month=3, end_year=2013, end_month=3):
    '''Creates a chart of the amount of imagery over a range of dates.
    :param start_year: The year to start the chart from.
    :param start_month: The month to start the chart from.
    :param end_year: The year to end the chart at.
    :param end_month: The month to end the chart at.
    '''
    # Get the data.
    df = get_data_from_date_range(start_year, start_month, end_year, end_month)

    # Create a dictionary showing the amount of imagery by date.
    dates_dict = {}

    # Populate the dictionary with the amount of imagery by date.
    for x in df.toPandas()["start-date"]:
        month = x.split("-")[1]
        year = x.split("-")[0]

        # If the date is already in the dictionary, increment the value. Otherwise, add the date to the dictionary.
        if month + "-" + year in dates_dict:
            dates_dict[month + "-" + year] += 1
        else:
            dates_dict[month + "-" + year] = 1

    # Create a chart of the data.
    plt.plot(list(dates_dict.keys()), list(dates_dict.values()), marker='o')

    # Add labels to the chart.
    for x, y in zip(list(dates_dict.keys()), list(dates_dict.values())):
        plt.text(x, y, str(y), ha='center', va='bottom')

    # Display the chart.
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Demo

# COMMAND ----------

chart_imagery_by_date(2010, 1, 2010, 12)
