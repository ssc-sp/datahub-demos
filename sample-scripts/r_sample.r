# Databricks notebook source
# Load the SparkR library
library(SparkR)

# Initialize SparkR session
sparkR.session(appName = "DemoSession", sparkConfig = list(spark.executor.instances = "2"))

# Create a Spark DataFrame
df <- createDataFrame(data.frame(Name = c("John", "Doe", "Jane", "Smith"),
                                 Age = c(28, 34, 23, 45),
                                 Department = c("Finance", "Marketing", "IT", "HR")))

# Show the DataFrame
showDF(df)

# Register the DataFrame as a temporary view
createOrReplaceTempView(df, "employees")

# Run a SQL query
result <- sql("SELECT Department, AVG(Age) as AverageAge FROM employees GROUP BY Department")

# Show the result of the SQL query
showDF(result)
