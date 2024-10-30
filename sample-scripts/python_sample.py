# Databricks notebook source
from pyspark.sql import functions as F

df = spark.createDataFrame([
    ("James", "Smith", "M", 30, "Doctor"),
    ("Anna", "Rose", "F", 41, "Engineer"),
    ("Robert", "Williams", "M", 62, "Lawyer")
], ["first_name", "last_name", "gender", "age", "job"])

# Display the DataFrame
display(df)

# Perform a simple transformation
df_upper = df.withColumn("name_uppercase", F.upper(F.col("first_name")))

# Display the transformed DataFrame
display(df_upper)

# Aggregate data
df_grouped = df.groupBy("job").count()

# Display the aggregated DataFrame
display(df_grouped)
