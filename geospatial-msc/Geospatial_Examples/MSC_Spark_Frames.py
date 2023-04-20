# Databricks notebook source
# MAGIC %md
# MAGIC <h2>Accessing ECCC Meteorological Data</h2>
# MAGIC <p>To access the data we want to collect the data from the MSC open data api portal: <a href='https://eccc-msc.github.io/open-data/msc-geomet/readme_en/' target='_blank'>Open Data</a></p>

# COMMAND ----------

# import required libraries
import requests
import pandas as pd
import io
import matplotlib
import numpy as np

# COMMAND ----------

# Obtain Data through API
weather_url = "https://api.weather.gc.ca/"
current = 0
lst = []
while current <= 200000:
    try:
        resp = requests.get(f'https://api.weather.gc.ca/collections/climate-hourly/items?f=csv&lang=en-CA&limit=10000&startindex={current}')
        r = resp.content
        df = pd.read_csv(io.StringIO(r.decode('utf-8')))
        lst.append(df)
        current += 10000
    except Exception as e:
        print(e)
        break
df = pd.concat(lst, axis=0, ignore_index=True)

# COMMAND ----------

# Transform pandas DF to pyspark
# note: this step can be skipped and spark DF can be created from original API response
pysparkdf = spark.createDataFrame(df)
pysparkdf.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC <h4>Testing Dataframe Performance</h4>
# MAGIC <p>Now that we have both a regular dataframe as well as a spark dataframe, we can use UDF's to test running speed of datafranes. Since spark dataframes uses parallel computation, for small files such as the one used here will be slower. But for larger files, the difference is more notable.</p>

# COMMAND ----------

# sample test code iterating a pandas dataframe
import datetime
for row in df.LOCAL_DATE:
    dt = datetime.datetime.strptime(row, '%Y-%m-%d %H:%M:%S')
    


# COMMAND ----------

# sample test code iterating a spark dataframe
for row in pysparkdf.collect():
    dt = datetime.datetime.strptime(row['LOCAL_DATE'], '%Y-%m-%d %H:%M:%S')

# COMMAND ----------

# MAGIC %md
# MAGIC <h4>Creating Temporary Delta Tables and Querying</h4>
# MAGIC We can also move the data into a delta table for faster queries

# COMMAND ----------

pysparkdf.createOrReplaceTempView("test_csv")

# COMMAND ----------

# MAGIC %sql
# MAGIC select STATION_NAME, x, y, TEMP, LOCAL_DATE from test_csv
# MAGIC where STATION_NAME != 'COMOX A' 
# MAGIC sort by TEMP DESC

# COMMAND ----------


