# Databricks notebook source
# MAGIC %md
# MAGIC <h2>Hydrometry data using GeoMet-OGC Open API</h2>

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>Use Case Scenario</h3> 
# MAGIC 
# MAGIC Cathy runs a boat tours and water sports company in Chilliwack, British-Columbia. She would like to track the water levels of a few hydrometric stations near her city to guide the management of the company's activities. She would like to have:
# MAGIC 
# MAGIC * The daily mean water levels of a three months period for all hydrometric stations located within 100 km of the city of Chilliwack
# MAGIC * Graphic and tabular views that present the daily mean water levels of that three months period
# MAGIC * An interactive map that displays the hydrometric stations located within 100 km of the city of Chilliwack and the most recent daily mean water level of these stations in the determined time period

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>Install and Import packages</h3>

# COMMAND ----------

# install the wheels for GDAL and osgeo
%pip install https://manthey.github.io/large_image_wheels/GDAL-3.5.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl#sha256=9387d6f4a71a132a7c5a13426a2491e9aded5e0974cadb43b9d579fac92541f8
# install folium library for mapping our results
%pip install branca
%pip install jinja2
%pip install folium
#install oswl and tabulate for WMS data
%pip install tabulate
%pip install OWSLib

# COMMAND ----------

# Modules importation
from datetime import date
import json
import math
from textwrap import fill

from matplotlib import pyplot as plt, dates as mdates
from osgeo import ogr, osr
from owslib.ogcapi.features import Features
import numpy as np
import pandas as pd
from tabulate import tabulate
import folium
from functools import reduce

from pyspark.sql.types import *
from pyspark.sql.functions import min, max
from pyspark.sql.functions import col
from pyspark.sql import DataFrame

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>Initiate parameters for the API request</h3>

# COMMAND ----------

# Parameters

# Coordinates of Chilliwack
lat = 49.162676
long = -121.958943

# Buffer size in kilometres
buffer = 100

# Start and end of the time period for which the data will be retrieved
start_date = date(2015, 6, 1)
end_date = date(2021, 8, 31)

# ESPG code of the preferred projection to create the buffer
# NAD83 / Statistics Canada Lambert
projection = 3347

# COMMAND ----------

# Parameters formatting for the OGC API - Features request

# Bounding box a little bigger than buffer size

# The buffer needs to be transformed in degrees to get
# the coordinates of the corners of the bounding box:
# Latitude: 1 km ≈ 0.009° 
# Longitude (at the 49th parallel): 1 km ≈ 0.014°
bbox = [
    long - buffer * 0.02,
    lat - buffer * 0.01,
    long + buffer * 0.02,
    lat + buffer * 0.01,
]

# Formatting of the selected time period
time_ = f"{start_date}/{end_date}"

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>Request Data and use GDAL to select user buffer</h3>

# COMMAND ----------

# Retrieval of hydrometric stations data
oafeat = Features("https://api.weather.gc.ca/")
station_data = oafeat.collection_items(
    "hydrometric-stations", bbox=bbox, STATUS_EN="Active"
)

# Verification of the retrieved data
if "features" in station_data:
    station_data = json.dumps(station_data, indent=4)
else:
    raise ValueError(
        "No hydrometric stations were found. Please verify the coordinates."
    )

# COMMAND ----------

# List of stations located inside the buffer zone

# Accessing the hydrometric stations layer
driver = ogr.GetDriverByName("GeoJSON")
data_source = driver.Open(station_data, 0)
layer = data_source.GetLayer()

# Identification of the input spatial reference system (SRS)
SRS_input = layer.GetSpatialRef()
SR = osr.SpatialReference(str(SRS_input))
epsg = SR.GetAuthorityCode(None)
SRS_input.ImportFromEPSG(int(epsg))

# Definition of the SRS used to project data
SRS_projected = osr.SpatialReference()
SRS_projected.ImportFromEPSG(projection)

# Transformation from input SRS to the prefered projection
transform = osr.CoordinateTransformation(SRS_input, SRS_projected)

# Creation of a buffer to select stations
point = ogr.Geometry(ogr.wkbPoint)
point.AddPoint(long, lat)
point.Transform(transform)
point_buffer = point.Buffer(buffer * 100000)  # The value must be in meters

# Selection of the stations in the buffer zone
stations = []

for feature in layer:
    geom = feature.GetGeometryRef().Clone()
    geom.Transform(transform)
    if geom.Intersects(point_buffer):
        stations.append(feature.STATION_NUMBER)

# Raising an error if no station were found
if not stations:
    raise ValueError(
        f"There are no hydrometric stations within {buffer} km"
        + " of the chosen coordinates. Please verify the coordinates."
    )

# COMMAND ----------

# This step is used to reduce how much the dataframe is split up to increase run time
sqlContext.sql("set spark.sql.shuffle.partitions=10")

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>Parse Data to remove empty stations and place within Dataframe</h3>
# MAGIC <p>Note that there are three cells which preform the same </p>

# COMMAND ----------

# This is the spark script which uses a single spark dataframe
# Retrieval of hydrometric data for each station

# Dictionary that will contain a data frame for each station with
# the historical daily mean water levels for the time period
hydrometric_data = []

# List of stations with no water level data
stations_without_data = []

historical_data_df = False

# Data retrieval and creation of the data frames
for station in stations:

    # Retrieval of water level data
    hydro_data = oafeat.collection_items(
        "hydrometric-daily-mean",
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        STATION_NUMBER=station,
    )
    # Creation of a data frame if there is data for the chosen time period
    if hydro_data["features"]:
        # if the dataframe hasn't been made, create it with the necessary features
        historical_data_format = [
            {
                "LATITUDE": el["geometry"]["coordinates"][1],
                "LONGITUDE": el["geometry"]["coordinates"][0],
                **el["properties"],
            }
            for el in hydro_data["features"]
        ]
        schema = StructType([StructField("STATION_NUMBER", StringType(), True), StructField("STATION_NAME", StringType(), True), StructField("DATE", StringType(), True), StructField("LEVEL", FloatType(), True), StructField("LATITUDE", FloatType(), True), StructField("LONGITUDE", FloatType(), True)])
        # Creation of the data frame
        historical_data_df = pd.DataFrame(
            historical_data_format,
            columns=[
                "STATION_NUMBER",
                "STATION_NAME",
                "DATE",
                "LEVEL",
                "LATITUDE",
                "LONGITUDE",
            ],
        )
        historical_data_df = historical_data_df.fillna(value=np.nan)
        # Adding the data frame to the hydrometric data dictionary
        if not historical_data_df["LEVEL"].isnull().all():
            # Removing any rows without water level data at the
            # end of the data frame
            while np.isnan(historical_data_df["LEVEL"].iloc[-1]):
                historical_data_df = historical_data_df.drop(
                    historical_data_df.tail(1).index
                )
            # Creating an index with the date in a datetime format
            '''
            historical_data_df["DATE"] = pd.to_datetime(
                historical_data_df["DATE"]
            )
            historical_data_df.set_index(["DATE"], inplace=True, drop=True)
            historical_data_df.index = historical_data_df.index.date
            '''
            # Adding the data frame to the dictionary
            spark_df = spark.createDataFrame(historical_data_df)
            hydrometric_data.append(spark_df)
        # If all the data is NaN, the station will be removed from the dataset
        else:
            stations_without_data.append(station)
    # If there is no data for the chosen time period, the station
    # will be removed from the dataset
    else:
        stations_without_data.append(station)

hydrometric_data_df = reduce(DataFrame.unionAll, hydrometric_data)
        
# Removing hydrometric stations without water level data from the station list
for station in stations_without_data:
    stations.remove(station)

# Raising an error if no station is left in the list
if not stations:
    raise ValueError(
        f"No water level data is available in the this {num_months}"
        + " months period for the selected hydrometric stations."
    )

# COMMAND ----------

hydrometric_data_df.show()

# COMMAND ----------

hydrometric_data_df.createOrReplaceTempView("tables")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DISTINCT STATION_NUMBER 
# MAGIC FROM tables

# COMMAND ----------

# Creation of an interactive plot with Matplotlib

# Hydrometric station to display on the plot
station_displayed_p = stations[1]

# Function to create a plot for the chosen hydrometric station
def interactive_plot(station):
    # Adjusting font and figure size
    params = {
        "legend.fontsize": "14",
        "figure.figsize": (9, 5),
        "axes.labelsize": "14",
        "axes.titlesize": "16",
        "xtick.labelsize": "12",
        "ytick.labelsize": "12",
    }
    plt.rcParams.update(params)
    
    # Creation of the plot
    fig, ax = plt.subplots()
    filtered_df = hydrometric_data_df.filter(hydrometric_data_df.STATION_NUMBER == station)
    line, = plt.plot(
        filtered_df.select('DATE').rdd.flatMap(lambda x: x).collect(),
        filtered_df.select('LEVEL').rdd.flatMap(lambda x: x).collect(),
        marker="o",
        label="Daily mean",
    )
    plt.legend()
    plt.grid(True, which="both")
    ax.set_title(
        fill(
            "Water levels at station {} ({})".format(
                hydrometric_data_df.filter(hydrometric_data_df.STATION_NUMBER == station).select("STATION_NAME")[0], station
            ), 60
        )
    )
    ax.set_ylabel("Water level (m)")
    ax.set_xlabel("Date")

    # Modification of the x axis ticks and tick labels
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # Creation of the annotations to display on hover
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(-60, -40),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    annot.set_visible(False)

    return line, annot, ax, fig


# Choosing the hydrometric stations to create the plot
line, annot, ax, fig = interactive_plot(station_displayed_p)


# Updating the annotation with the data point information
def update_annot(ind):
    # Identifying the annotation to display
    x, y = line.get_data()
    annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
    
    # Adding text to the annotation (date and water level)
    date_x = x[ind["ind"]][0]
    level_y = round(y[ind["ind"]][0], 2)
    text = "{}\nDaily mean: {} m".format(date_x, level_y)
    annot.set_text(text)
    
    # Setting annotation transparency
    annot.get_bbox_patch().set_alpha(0.8)


# Display of annotations on hover
def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = line.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()


# Adding the feature that displays annotations on hover               
fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()

# COMMAND ----------

# Creation of the table

# Hydrometric station to display in the table
station_displayed_t = stations[1]

# Option to show all rows
pd.options.display.max_rows


# Function to show the table of the chosen hydrometric station
def data_table(station):
    # Print table title
    print(
        "Water levels at station "
        + f"{hydrometric_data[station]['STATION_NAME'][0]}"
        + f" ({station})"
    )
    # Selecting the desired columns and changing the columns names
    displayed_df = hydrometric_data[station][["LEVEL"]].round(2).rename(
        columns={
            "LEVEL": "Water level daily mean (m)",
        }
    )
    displayed_df.index = displayed_df.index.rename("Date")
    return displayed_df

hydrometric_data[station_displayed_t].select(['DATE', 'LEVEL']).show()
'''
print(tabulate(data_table(station_displayed_t),
               headers='keys',
               tablefmt='pretty'))
'''

# COMMAND ----------

from pyspark.sql.functions import col
labels = []
all_lat = []
all_lon = []
for station in stations:
    latest_data = hydrometric_data_df.filter(hydrometric_data_df.STATION_NUMBER == station).tail(1)[0]
    labels.append(
        f"{latest_data[1]}\n"
        + f"Station ID: {latest_data[0]}\n"
        + f"Date: {latest_data[2]}\n"
        + f"Water level: {round(latest_data[3], 2)} m"
    )
    all_lat.append(latest_data.LATITUDE)
    all_lon.append(latest_data.LONGITUDE)

annotations = [None for label in labels]
m = folium.Map(location=[lat, long])
feature_group = folium.FeatureGroup("Locations")
for lt, lng, name in zip(all_lat, all_lon, labels):
    feature_group.add_child(folium.Marker(location=[lt,lng],popup=name))
m.add_child(feature_group)
m

# COMMAND ----------

m.save('map.html')

# COMMAND ----------

# MAGIC %md
# MAGIC <h4>Spark DF No Optimization</h4>
# MAGIC command 7 = 4.23
# MAGIC command 8 = 30.96
# MAGIC command 9 = 0.45
# MAGIC command 10 = 0.13 
# MAGIC command 11 = 2.82 
# MAGIC 
# MAGIC <h4>Pandas DF</h4>
# MAGIC command 7 = 0.04
# MAGIC command 8 = 4.74
# MAGIC command 9 = 0.3
# MAGIC command 10 = 0.03
# MAGIC command 11 = 0.09
