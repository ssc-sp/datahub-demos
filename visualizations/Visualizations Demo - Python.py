# Databricks notebook source
# MAGIC %md
# MAGIC # Data Visualizations in Databricks - Python
# MAGIC
# MAGIC This notebook demonstrates how to create and view visualizations of your Databricks data. This notebook uses Python, while another notebook shows some similar code in R.
# MAGIC
# MAGIC ## Titanic Passenger Dataset
# MAGIC
# MAGIC This dataset contains information about the passengers aboard the Titanic, such as their age, fare paid, and whether they survived. It's often used for learning AI, since you can use this information to predict whether a passengers survived or not, but it also provides some interesting data to visualize.
# MAGIC
# MAGIC ### 1. Loading Data
# MAGIC
# MAGIC We load our data from the mounted storage directly from the Federal Science DataHub.

# COMMAND ----------

df = spark.read.option("header","true").csv("/mnt/fsdh-dbk-main-mount/Sean/titanic_train.csv")
display(df)

# COMMAND ----------

import pandas as pd

df_pandas = df.toPandas()
display(df)

# COMMAND ----------

print("Hello world!")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 2. Data Visualizations
# MAGIC
# MAGIC We display some visualizations about our data.
# MAGIC
# MAGIC #### 2.1. Pie Chart - Survival Rate of Passengers
# MAGIC
# MAGIC We create several pie charts to evaluate the survival rate of passengers depending on what class they were in, or what their gender is.
# MAGIC
# MAGIC 1 indicates they survived Titanic, while 0 indicates they were no longer alive.

# COMMAND ----------

display(df_pandas)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2. Bar Plot - Gender by Class

# COMMAND ----------

display(df_pandas)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3. Box and Scatter Plot - Fare Paid per Class
# MAGIC
# MAGIC We can use either a box or scatter plot to depict how much people were paying based on class, allowing us to see the maximums and minimums, as well as the medians.

# COMMAND ----------

df_pandas = df_pandas.astype({'Fare':'float'})
display(df_pandas)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### 2.4. Boarding Location by Class
# MAGIC
# MAGIC We create a heatmap to illustrate where passengers were boarding from. The heatmap is based on (a) Class, either 1st, 2nd, or 3rd, and (b) Embarkation Location, either Southampton, England, Cherbourg, France, or Queenstown, Ireland. We can observe the largest amount of people boarded in Southampton and the fewest in Queenstown.

# COMMAND ----------

df_pandas = df_pandas.astype({'Age':'float'})
display(df_pandas)

# COMMAND ----------

# MAGIC %md
# MAGIC ## IMDB Movie Review Dataset
# MAGIC
# MAGIC This dataset contains a set of 50,000 movie reviews from IMDB and the associated sentiment. It's popular for training sentiment analysis models.
# MAGIC
# MAGIC ### 1. Loading the Data
# MAGIC
# MAGIC We load the data from our DataHub storage again.

# COMMAND ----------

df = spark.read.option("header","true").csv("/mnt/fsdh-dbk-main-mount/Sean/imdb_dataset_preprocessed.csv")
display(df)

# COMMAND ----------

import pandas as pd

df_pandas = df.toPandas()
display(df_pandas)

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC
# MAGIC ### 2. Data Visualizations
# MAGIC
# MAGIC We display some visualizations about our data.
# MAGIC
# MAGIC #### 2.1. Pie/Bar Chart - Split of Positive vs Negative Reviews
# MAGIC
# MAGIC We create several a pie and a bar chart to evaluate how many reviews express a positive and negative sentiment

# COMMAND ----------

display(df_pandas)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2. Word Map
# MAGIC
# MAGIC We create a word map, which illustrates some of the most common sentiments expressed in a corpus of texts.

# COMMAND ----------

df_pandas = df_pandas.head(2500)
display(df_pandas)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Maps
# MAGIC
# MAGIC Lastly, we have some map visualization tools to demonstrate.
# MAGIC
# MAGIC ### 1. Load Data for Choropleth Map
# MAGIC
# MAGIC Choropleth maps rely on geographic localities, such as countries or states. We can colour them based on values. We will demonstrate this using some COVID data from Our World in Data.

# COMMAND ----------

df = spark.read.option("header","true").csv("/mnt/fsdh-dbk-main-mount/Sean/owid-covid-data.csv")
display(df)

# COMMAND ----------

import pandas as pd

df_pandas = df.toPandas()
display(df_pandas)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 2. Example Choropleth Map + Counter

# COMMAND ----------

df_pandas = df_pandas.astype({'total_cases':'float'})
display(df_pandas)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Create Data for Marker Map
# MAGIC
# MAGIC A marker is placed at a set of coordinates on the map. The query result must return latitude and longitude pairs.
# MAGIC
# MAGIC In our case, we will simply create some coordinates for these points.

# COMMAND ----------

import random

def generate_coordinates():
    latitudes = []
    longitudes = []
    
    for _ in range(10):
        latitude = random.uniform(-90, 90)
        longitude = random.uniform(-180, 180)
        latitudes.append(latitude)
        longitudes.append(longitude)
    
    return latitudes, longitudes

# Generating coordinates
generated_latitudes_1, generated_longitudes_1 = generate_coordinates()

# Printing the results
print("Generated Latitudes:", generated_latitudes_1)
print("Generated Longitudes:", generated_longitudes_1)

# Generating coordinates
generated_latitudes_2, generated_longitudes_2 = generate_coordinates()

# Printing the results
print("Generated Latitudes:", generated_latitudes_2)
print("Generated Longitudes:", generated_longitudes_2)

data = {"Latitude": generated_latitudes_1 + generated_latitudes_2, "Longitude": generated_longitudes_1 + generated_longitudes_2, "Type": ['b','b','b','b','b','b','b','b','b','b','r','r','r','r','r','r','r','r','r','r']}
df = pd.DataFrame(data)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 4. Example Marker Map

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## RMS Titanic Passenger Dataset
# MAGIC
# MAGIC This dataset contains information about the passengers aboard the Titanic, such as their age, fare paid, and whether they survived. It's often used for learning AI, since you can use this information to predict whether a passengers survived or not, but it also provides some interesting data to visualize.
