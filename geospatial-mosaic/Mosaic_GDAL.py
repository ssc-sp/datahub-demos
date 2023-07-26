# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Mosaic GDAL Demo
# MAGIC
# MAGIC This notebook is intended to show the setup and usage of GDAL using Mosaic.
# MAGIC
# MAGIC Full installation guide can be found at https://github.com/ssc-sp/datahub-docs/blob/main/UserGuide/Workspace/geospatial-workload.md
# MAGIC
# MAGIC ## 1. Installation
# MAGIC
# MAGIC Run the following ONLY ONCE, then edit the cluster to add `dbfs:/FileStore/geospatial/mosaic/gdal/mosaic-gdal-init.sh` to the init, then restart the cluster.

# COMMAND ----------

import mosaic as mos
mos.enable_mosaic(spark, dbutils)
mos.setup_gdal(spark)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2. Setup
# MAGIC
# MAGIC Enables Mosaic and GDAL. If successful, it will output "GDAL enabled." and the version number.

# COMMAND ----------

import mosaic as mos
mos.enable_mosaic(spark, dbutils)
mos.enable_gdal(spark)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Raster Format Reading
# MAGIC
# MAGIC This code allows you to read a GeoTIFF file. In our example, we use an image from RADARSAT-1.

# COMMAND ----------

df = spark.read.format("gdal").option("driverName", "GTiff").load('/FileStore/shared_uploads/sean.stilwell@ssc-spc.gc.ca/RS1_X0493228_SCWA_20051102_201449_HH_SCW__1_-1.tif')
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Raster Functions
# MAGIC
# MAGIC The library gives various example Raster Functions. We use them on a sample image from the RADARSAT-1 Open Data repo (https://registry.opendata.aws/radarsat-1/)
# MAGIC
# MAGIC ### Band Metadata

# COMMAND ----------

from pyspark.sql import functions as F

tiff = '/FileStore/shared_uploads/sean.stilwell@ssc-spc.gc.ca/RS1_X0493228_SCWA_20051102_201449_HH_SCW__1_-1.tif'

df = spark.read.format("gdal").option("driverName", "GTiff").load(tiff)

df.select(mos.rst_bandmetadata('path', F.lit(1))).display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Georeference

# COMMAND ----------

df.select(mos.rst_georeference("path")).limit(1).display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Height

# COMMAND ----------

df.select(mos.rst_height('path')).display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Memory Size

# COMMAND ----------

df.select(mos.rst_memsize('path')).display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Metadata
# MAGIC
# MAGIC The R1 images have a large metadata that provides additional info.

# COMMAND ----------

df.select(mos.rst_metadata('path')).display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Raster to Grid Functions
# MAGIC
# MAGIC These require a lot more memory than other functions.

# COMMAND ----------

df.select(mos.rst_rastertogridavg('path', F.lit(3))).display()

# COMMAND ----------

df.select(mos.rst_rastertogridmedian('path', F.lit(3))).display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Raster to World Coordinates

# COMMAND ----------

df.select(mos.rst_rastertoworldcoord('path', F.lit(2), F.lit(2))).display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Rotation

# COMMAND ----------

df.select(mos.rst_rotation('path')).display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Scaley

# COMMAND ----------

df.select(mos.rst_scaley('path')).display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Summary

# COMMAND ----------

df.select(mos.rst_summary('path')).display()
