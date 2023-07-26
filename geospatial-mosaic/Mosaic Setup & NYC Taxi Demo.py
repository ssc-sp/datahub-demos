# Databricks notebook source
# MAGIC %md
# MAGIC ## Prerequisite
# MAGIC
# MAGIC In your cluster, add the "databricks-mosaic" package from PyPi under the Libraries tab.
# MAGIC
# MAGIC ## 1. Setup
# MAGIC
# MAGIC This code will generate an init script for your Databricks cluster for Mosaic. You only need to run this once.
# MAGIC
# MAGIC **Note:** You must use ML Runtimes 11 or 12. 10 (or below) and 13 are not supported.

# COMMAND ----------

import mosaic as mos
mos.enable_mosaic(spark, dbutils)
mos.setup_gdal(spark)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2. Enable GDAL

# COMMAND ----------

import mosaic as mos

mos.enable_mosaic(spark, dbutils)
mos.enable_gdal(spark)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3. NYC Taxi Dataset
# MAGIC
# MAGIC This is an example of performing spatial point-in-polygon joins on the NYC Taxi dataset
# MAGIC
# MAGIC ### 3.1. Download the NYC Taxi Dataset

# COMMAND ----------

import requests
import pathlib

user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

raw_path = f"dbfs:/tmp/mosaic/{user_name}"
raw_taxi_zones_path = f"{raw_path}/taxi_zones"

print(f"The raw data will be stored in {raw_path}")

taxi_zones_url = 'https://data.cityofnewyork.us/api/geospatial/d3c5-ddgc?method=export&format=GeoJSON'

# The DBFS file system is mounted under /dbfs/ directory on Databricks cluster nodes

local_taxi_zones_path = pathlib.Path(raw_taxi_zones_path.replace('dbfs:/', '/dbfs/'))
local_taxi_zones_path.mkdir(parents=True, exist_ok=True)

req = requests.get(taxi_zones_url)
with open(local_taxi_zones_path / f'nyc_taxi_zones.geojson', 'wb') as f:
  f.write(req.content)

# COMMAND ----------

display(dbutils.fs.ls(raw_taxi_zones_path))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 3.2. Spatial point-in-polygon joins on the NYC Taxi dataset

# COMMAND ----------

user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

raw_path = f"dbfs:/tmp/mosaic/{user_name}"
raw_taxi_zones_path = f"{raw_path}/taxi_zones"

print(f"The raw data is stored in {raw_path}")

# COMMAND ----------

from pyspark.sql.functions import *
import mosaic as mos
mos.enable_mosaic(spark, dbutils)

neighbourhoods = (
  spark.read
    .option("multiline", "true")
    .format("json")
    .load(raw_taxi_zones_path)
    .select("type", explode(col("features")).alias("feature"))
    .select("type", col("feature.properties").alias("properties"), to_json(col("feature.geometry")).alias("json_geometry"))
    .withColumn("geometry", mos.st_aswkt(mos.st_geomfromgeojson("json_geometry")))
)

display(
  neighbourhoods
)

# COMMAND ----------

display(
  neighbourhoods
    .withColumn("calculatedArea", mos.st_area(col("geometry")))
    .withColumn("calculatedLength", mos.st_length(col("geometry")))
    # Note: The unit of measure of the area and length depends on the CRS used.
    # For GPS locations it will be square radians and radians
    .select("geometry", "calculatedArea", "calculatedLength")
)

# COMMAND ----------

tripsTable = spark.table("delta.`/databricks-datasets/nyctaxi/tables/nyctaxi_yellow`")

trips = (
  tripsTable
    .drop("vendorId", "rateCodeId", "store_and_fwd_flag", "payment_type")
    .withColumn("pickup_geom", mos.st_astext(mos.st_point(col("pickup_longitude"), col("pickup_latitude"))))
    .withColumn("dropoff_geom", mos.st_astext(mos.st_point(col("dropoff_longitude"), col("dropoff_latitude"))))
)

display(trips.select("pickup_geom", "dropoff_geom"))

# COMMAND ----------

from mosaic import MosaicFrame

neighbourhoods_mosaic_frame = MosaicFrame(neighbourhoods, "geometry")
optimal_resolution = neighbourhoods_mosaic_frame.get_optimal_resolution(sample_fraction=0.75)

print(f"Optimal resolution is {optimal_resolution}")

# COMMAND ----------

display(
  neighbourhoods_mosaic_frame.get_resolution_metrics(sample_rows=150)
)

# COMMAND ----------

tripsWithIndex = (trips
  .withColumn("pickup_h3", mos.grid_pointascellid(col("pickup_geom"), lit(optimal_resolution)))
  .withColumn("dropoff_h3", mos.grid_pointascellid(col("dropoff_geom"), lit(optimal_resolution)))
)

display(tripsWithIndex)

# COMMAND ----------

neighbourhoodsWithIndex = (neighbourhoods

                           # We break down the original geometry in multiple smaller mosaic chips, each with its
                           # own index
                           .withColumn("mosaic_index", mos.grid_tessellateexplode(col("geometry"), lit(optimal_resolution)))

                           # We don't need the original geometry any more, since we have broken it down into
                           # Smaller mosaic chips.
                           .drop("json_geometry", "geometry")
                          )
                          
display(neighbourhoodsWithIndex)

# COMMAND ----------

pickupNeighbourhoods = neighbourhoodsWithIndex.select(col("properties.zone").alias("pickup_zone"), col("mosaic_index"))

withPickupZone = (
  tripsWithIndex.join(
    pickupNeighbourhoods,
    tripsWithIndex["pickup_h3"] == pickupNeighbourhoods["mosaic_index.index_id"]
  ).where(
    # If the borough is a core chip (the chip is fully contained within the geometry), then we do not need
    # to perform any intersection, because any point matching the same index will certainly be contained in
    # the borough. Otherwise we need to perform an st_contains operation on the chip geometry.
    col("mosaic_index.is_core") | mos.st_contains(col("mosaic_index.wkb"), col("pickup_geom"))
  ).select(
    "trip_distance", "pickup_geom", "pickup_zone", "dropoff_geom", "pickup_h3", "dropoff_h3"
  )
)

display(withPickupZone)

# COMMAND ----------

dropoffNeighbourhoods = neighbourhoodsWithIndex.select(col("properties.zone").alias("dropoff_zone"), col("mosaic_index"))

withDropoffZone = (
  withPickupZone.join(
    dropoffNeighbourhoods,
    withPickupZone["dropoff_h3"] == dropoffNeighbourhoods["mosaic_index.index_id"]
  ).where(
    col("mosaic_index.is_core") | mos.st_contains(col("mosaic_index.wkb"), col("dropoff_geom"))
  ).select(
    "trip_distance", "pickup_geom", "pickup_zone", "dropoff_geom", "dropoff_zone", "pickup_h3", "dropoff_h3"
  )
  .withColumn("trip_line", mos.st_astext(mos.st_makeline(array(mos.st_geomfromwkt(col("pickup_geom")), mos.st_geomfromwkt(col("dropoff_geom"))))))
)

display(withDropoffZone)

# COMMAND ----------


