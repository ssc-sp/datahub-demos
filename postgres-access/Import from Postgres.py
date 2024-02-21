# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Load data from Postgres to Delta Lake
# MAGIC
# MAGIC This notebook shows you how to import data from JDBC Postgres databases into a Delta Lake table using Python.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Step 1: Connection information
# MAGIC
# MAGIC First define some variables to programmatically create these connections.
# MAGIC
# MAGIC Replace all the variables in angle brackets `<>` below with the corresponding information.

# COMMAND ----------

driver = "org.postgresql.Driver"

database_host = "host"
database_port = "port" # update if you use a non-default port
database_name = "name" # eg. postgres
table = "table" # if your table is in a non-default schema, set as <schema>.<table-name> 
user = "user"
password = "password"

url = f"jdbc:postgresql://{database_host}:{database_port}/{database_name}"

print(url)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The full URL printed out above should look something like:
# MAGIC
# MAGIC ```
# MAGIC jdbc:postgresql://localhost:5432/my_database
# MAGIC ```
# MAGIC
# MAGIC ### Check connectivity
# MAGIC
# MAGIC Depending on security settings for your Postgres database and Databricks workspace, you may not have the proper ports open to connect.
# MAGIC
# MAGIC Replace `<database-host-url>` with the universal locator for your Postgres implementation. If you are using a non-default port, also update the 5432.
# MAGIC
# MAGIC Run the cell below to confirm Databricks can reach your Postgres database.

# COMMAND ----------

# MAGIC %sh
# MAGIC nc -vz "host" 5432

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Step 2: Reading the data
# MAGIC
# MAGIC Now that you've specified the file metadata, you can create a DataFrame. Use an *option* to infer the data schema from the file. You can also explicitly set this to a particular schema if you have one already.
# MAGIC
# MAGIC First, create a DataFrame in Python, referencing the variables defined above.

# COMMAND ----------

remote_table = (spark.read
    .format("jdbc")
    .option("driver", driver)
    .option("url", url)
    .option("dbtable", table)
    .option("user", user)
    .option("password", password)
    .load()
)

# COMMAND ----------

# MAGIC %md
# MAGIC You can view the results of this remote table query.

# COMMAND ----------

display(remote_table)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Step 3: Create a Delta table
# MAGIC
# MAGIC The DataFrame defined and displayed above is a temporary connection to the remote database.
# MAGIC
# MAGIC To ensure that this data can be accessed by relevant users throughout your workspace, save it as a Delta Lake table using the code below.

# COMMAND ----------

target_table_name = "<target-schema>.<target-table-name>"
remote_table.write.mode("overwrite").saveAsTable(target_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC This table will persist across cluster sessions, notebooks, and personas throughout your organization.
# MAGIC
# MAGIC The code below demonstrates querying this data with Python and SQL.

# COMMAND ----------

display(spark.table(target_table_name))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM <target-schema>.<target-table-name>
