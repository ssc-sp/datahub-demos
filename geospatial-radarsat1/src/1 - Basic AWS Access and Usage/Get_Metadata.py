# Databricks notebook source
# MAGIC %md
# MAGIC # Get Metadata
# MAGIC 
# MAGIC These are a series of functions that allow you to access the metadata from the images in the S3 bucket.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup
# MAGIC 
# MAGIC We first have to run a pip install of geopy. You also need boto3 if it's not already installed.

# COMMAND ----------

# MAGIC %pip install geopy

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC 
# MAGIC Importing the necessary packages and setting up some variables we'll be needing to access the files.

# COMMAND ----------

import pandas as pd
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from geopy.geocoders import Nominatim
from geopy.point import Point

# Setting up access
BUCKET_NAME = 'radarsat-r1-l1-cog'
s3client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
paginator = s3client.get_paginator('list_objects_v2')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Functions
# MAGIC 
# MAGIC These are the two functions that allow us to retrieve the metadata

# COMMAND ----------

# MAGIC %md
# MAGIC ### get_data_from_month
# MAGIC 
# MAGIC This function retrieves the metadata for a single month of a single year.

# COMMAND ----------

def get_data_from_month(year = -1, month = -1, country_name=None, attribute_name=None, attribute_value=None, to_csv=False, printing=False):
    """Gets the data from the S3 bucket for a given month / year.

    :param year: Year of the data to be downloaded.
    :param month: Month of the data to be downloaded.
    :param country_name: Country name of the data to be downloaded.
    :param attribute_name: Name of the attribute to use when filtering the metadata.
    :param attribute_value: Value of the attribute to use when filtering the metadata.
    :param to_csv: Whether or not to convert the output to CSV.
    """

    # We limit our data by year or by month (if given)
    if month != -1:
        prefix = str(year) + "/" + str(month) + "/"
    elif year != -1:
        prefix = str(year) + "/"
    else:
        prefix = ""
        
    if printing:
        print(prefix)

    # We then set up our parameters to make the call
    parameters = {
        'Bucket': BUCKET_NAME,
        'Prefix': prefix
    }

    page_iterator = paginator.paginate(**parameters)

    # Variables needed for the script
    list = [] # A list of lists containing the metadata
    column_names = True # Only runs during first iteration

    # Iterates through the bucket
    for bucket in page_iterator:
        # If there is no data, we return an empty dataframe
        if not 'Contents' in bucket:
            print("Note: There is no imagery for year", year, "month", month)
            return pd.DataFrame()

        # Otherwise, we iterate through the contents
        for file in bucket['Contents']:
            try:
                metadata = s3client.head_object(Bucket='radarsat-r1-l1-cog', Key=file['Key'])

                # If an attribute requirement was given, we check it here.
                if attribute_name and attribute_value:
                    if metadata["Metadata"] and attribute_name in metadata["Metadata"]:
                        # If so, we check that the attribute value is contained in the metadata.
                        if metadata["Metadata"][attribute_name].lower() == attribute_value.lower():
                            pass
                        else: # If it doesn't match, we continue with the next file
                            continue
                    else: # If it's not in the metadata, we continue with the next file
                        continue

                # If a country name was given, we check it here.
                if country_name:
                    # We then have to use the metadata to get the longitude and latitude
                    test = metadata['Metadata']['scene-centre']
                    coords = test.split("(")[1].split(" ")
                    coords[1] = coords[1].replace(")", "")

                    # We use the geopy library to get the country name from the longitude and latitude
                    geolocator = Nominatim(user_agent="radarsat-r1-l1-cog")
                    location = geolocator.reverse(Point(coords[1], coords[0]), language='en')
                    if location: # If the location is not null, we can continue
                        if location.address: # Same with address being not null
                            # We can then parse the country, depending what the type of location address is
                            if type(location.address) is list:
                                country = location.address[len(location.address) - 1]
                            else:
                                address = location.address.split(",")
                                country = address[-1].strip()
                        # We then check if the country matches the country name and if it does, we retrieve metadata
                        if country.lower() == country_name.lower():
                            pass
                        else:
                            continue

                # During the first iteration, we create a list of columns using the metadata tags
                if column_names:
                    columns = []
                    for column in metadata["Metadata"]:
                        columns.append(column)
                    columns.append('download_link')
                    list.append(columns)
                    column_names = False
                temp = [None] * len(columns)

                # We then append the value of each metadata field
                for key, value in metadata["Metadata"].items():
                    for i in range(len(columns) - 1):
                        if columns[i] == key:
                            temp[i] = value
                            break
                temp[len(columns) - 1] = 'https://s3-ca-central-1.amazonaws.com/radarsat-r1-l1-cog/' + file['Key']
                list.append(temp)
                if (len(temp) != len(columns)):
                    print("Not equal!")
                
            except Exception as e:
                print(e)
                print("Failed {}".format(file['Key']))

    # We pop the first element of the list, which is the column names for the dataframe
    column_names = list.pop(0)

    # We can then look for errors in the data
#     df = pd.DataFrame(list, columns=column_names)
    df = spark.createDataFrame(data = list, schema = column_names)

    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ### get_data_from_date_range
# MAGIC 
# MAGIC This allows the user to give a start + end point for the metadata collection. Runs the preceding function and combines all the monthly metadata.

# COMMAND ----------

def get_data_from_date_range(start_year=1996, start_month=3, end_year=2013, end_month=3, country_name=None, attribute_name=None, attribute_value=None, to_csv=False):
    """Gets all the data from the S3 bucket for a given date range.

    :param start_year: Year of the start of the date range.
    :param start_month: Month of the start of the date range.
    :param end_year: Year of the end of the date range.
    :param end_month: Month of the end of the date range.
    :param country_name: Country name of the data to be downloaded.
    :param attribute_name: Name of the attribute to use when filtering the metadata.
    :param attribute_value: Value of the attribute to use when filtering the metadata.
    :param to_csv: Whether or not to convert the output to CSV.
    """

    # Correct potentially invalid parameters
    if start_year < 1996 or (start_year == 1996 and start_month < 3):
        start_year = 1996
        start_month = 3
    if end_year > 2013 or (end_year == 2013 and end_month > 3):
        end_year = 2013
        end_month = 3

    # Initialize an empty dataframe
    super_data = None

    # Iterate through the years
    while start_year <= end_year:
        # If we are in the same year, we iterate through the months
        if start_month > 12:
            start_year += 1
            start_month = 1
        # If we reach the end month, we break
        if start_year == end_year and start_month > end_month:
            break

        # We then get the data for the current month
        if not super_data or len(super_data.head(1)) <= 0: # If the dataframe is empty, we just start it with the data
            super_data = get_data_from_month(start_year, start_month, country_name, attribute_name, attribute_value, False)
        else: # Otherwise, we append the data for the current month
            data = get_data_from_month(start_year, start_month, country_name, attribute_name, attribute_value, False)
            super_data = super_data.union(data)
        start_month += 1
    if to_csv:
        super_data.write.csv("/tmp/spark_output/csv3")
    return super_data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo
# MAGIC 
# MAGIC We now do a brief demonstration of the function, collecting metadata from March to June of 2007 for images taken in Canada

# COMMAND ----------

output = get_data_from_date_range(2007, 3, 2007, 6, 'canada')
output.show()

# COMMAND ----------

output.toPandas()
