# Databricks notebook source
# MAGIC %md
# MAGIC # PostgreSQL Demo
# MAGIC
# MAGIC ## Step 1: Install Packages

# COMMAND ----------

!pip install psycopg2

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Set up Connection Details

# COMMAND ----------

HOST="YOUR_HOST"
DATABASE="YOUR_HOST"
USER="YOUR_USERNAME"
PASSWORD="YOUR_PASSWORD"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Connect to Database

# COMMAND ----------

import psycopg2
from psycopg2 import sql

conn = psycopg2.connect(
    host=HOST,
    database=DATABASE,
    user=USER,
    password=PASSWORD
)
cursor = conn.cursor()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Sample Operations
# MAGIC
# MAGIC ### Creating a Table

# COMMAND ----------

create_table_query = """
CREATE TABLE IF NOT EXISTS celestial_bodies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    body_type VARCHAR(50),
    mean_radius_km NUMERIC,
    mass_kg NUMERIC,
    distance_from_sun_km NUMERIC
);
"""
cursor.execute(create_table_query)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inserting Data

# COMMAND ----------

dummy_data = [
    ('Mercury', 'Planet', 2439.7, 3.3011e23, 57909227),
    ('Venus', 'Planet', 6051.8, 4.8675e24, 108209475),
    ('Earth', 'Planet', 6371.0, 5.97237e24, 149598262),
    ('Mars', 'Planet', 3389.5, 6.4171e23, 227943824),
    ('Jupiter', 'Planet', 69911, 1.8982e27, 778340821),
    ('Europa', 'Moon', 1560.8, 4.7998e22, 670900000), 
    ('Ganymede', 'Moon', 2634.1, 1.4819e23, 670900000),
    ('Ceres', 'Dwarf Planet', 473, 9.3835e20, 413700000),
    ('Pluto', 'Dwarf Planet', 1188.3, 1.303e22, 5906440628)
]

insert_query = """
INSERT INTO celestial_bodies (name, body_type, mean_radius_km, mass_kg, distance_from_sun_km) VALUES (%s, %s, %s, %s, %s);
"""
cursor.executemany(insert_query, dummy_data)
conn.commit()

# COMMAND ----------

conn.commit()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Retrieving Data

# COMMAND ----------

select_query = "SELECT * FROM celestial_bodies;"
cursor.execute(select_query)

# Fetch all the rows
rows = cursor.fetchall()
for row in rows:
    print(row)

# COMMAND ----------

import pandas as pd

# Load the data into a Pandas DataFrame
df = pd.read_sql_query(select_query, conn)

# Print the DataFrame
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Close the Connection

# COMMAND ----------

conn.close()
