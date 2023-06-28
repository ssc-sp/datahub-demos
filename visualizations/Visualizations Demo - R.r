# Databricks notebook source
# MAGIC %md
# MAGIC # Data Visualizations in Databricks - R
# MAGIC
# MAGIC This notebook demonstrates how to create and view visualizations of your Databricks data. This notebook uses R, while another notebook shows some similar code in Python.
# MAGIC
# MAGIC ## Titanic Passenger Dataset
# MAGIC
# MAGIC This dataset contains information about the passengers aboard the Titanic, such as their age, fare paid, and whether they survived. It's often used for learning AI, since you can use this information to predict whether a passengers survived or not, but it also provides some interesting data to visualize.
# MAGIC
# MAGIC ### 1. Loading Data
# MAGIC
# MAGIC We load our data from the mounted storage directly from the Federal Science DataHub.

# COMMAND ----------

library(SparkR)
sparkR.session()
df <- read.df("dbfs:/mnt/fsdh-dbk-main-mount/Sean/titanic_train.csv", source = "csv", header = T)
head(df, 3)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 2. Data Visualizations
# MAGIC
# MAGIC We display some visualizations about our data.
# MAGIC
# MAGIC #### 2.1. Pie Chart - Survival Rate of Passengers
# MAGIC
# MAGIC We create several pie charts to evaluate the survival rate of passengers depending on what class they were in.

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2. Bar Plot - Gender by Class

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3. Box and Scatter Plot - Fare Paid per Class
# MAGIC
# MAGIC We can use either a box or scatter plot to depict how much people were paying based on class, allowing us to see the maximums and minimums, as well as the medians.

# COMMAND ----------

display(df)
