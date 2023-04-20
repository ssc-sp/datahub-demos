# Databricks notebook source
# MAGIC %md
# MAGIC #Mounting the Bucket to DBFS
# MAGIC 
# MAGIC To start, we must mount the RADARSAT-1 bucket from Amazon Web Services to the local drive. We use the dbutils functionality to do this. For information about this data, visit: https://registry.opendata.aws/radarsat-1/
# MAGIC 
# MAGIC After mounting it, we display the contents of the mounted drive. It displays a set of folders for each year of RADARSAT-1's operation.

# COMMAND ----------

BUCKET_NAME = 'radarsat-r1-l1-cog' # Must not change, points to the S3 bucket on AWS
MOUNT_NAME = "radarsat_mount" # Can be changed to anything, local name of the mounted storage
MOUNT_PATH = "/mnt/%s" % MOUNT_NAME

# Check to ensure that r1 isn't already mounted. If not, mount it.
if ([item.mountPoint for item in dbutils.fs.mounts()].count(MOUNT_PATH) == 0):
    dbutils.fs.mount("s3a://%s" % BUCKET_NAME, "/mnt/%s" % MOUNT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC # Accessing the Mounted Drive
# MAGIC 
# MAGIC To demonstrate that it is now successfully mounted, we can list some files from it. For example, we will look at the imagery taken in August of 2002.

# COMMAND ----------

display(dbutils.fs.ls("/mnt/" + MOUNT_NAME + "/2002/8/"))

# COMMAND ----------

# MAGIC %md
# MAGIC We can also diplay an image from that year

# COMMAND ----------

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/dbfs/mnt/test_radarsat_mount/2007/8/RS1_M0620349_SCNA_20070825_020117_HH_SCN.tif')
plt.imshow(img)
