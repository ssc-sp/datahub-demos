import os
import zipfile
import subprocess

os.system('mkdir -p /fsdh/workdir')

#Make dictionary of directory names used by sen2cor
options = {'L1ADirName' : '/ccrs/batch/data/',
           'iDirName' : '/ccrs/batch/data/sen2cor/inputdir/', 
           'wDirName' : '/fsdh/workdir/',
           'oDirName' : '/ccrs/batch/data/sen2cor/outputdir2/'}
print(options.keys())

# Process all files in the L1A directory - results for 4 cores
# Currently we only process one manually and break the loop as we are benchmarking
for item in os.listdir(options['L1ADirName']):
    if item.endswith('zip'):
        with zipfile.ZipFile(options['L1ADirName']+item, 'r') as zip_ref:
            # zip_ref.extractall(options['iDirName'])
            iFileName = options['iDirName']+os.listdir(options['iDirName'])[0]
            print(iFileName)

            # hard coded to process first file for testing
            os.system('/bin/bash /fsdh/s2/sen2cor021100/bin/L2A_Process --output_dir /fsdh/workdir /ccrs/batch/data/sen2cor/inputdir/S2A_MSIL1C_20180101T160031_N0206_R054_T18TVQ_20180101T174126.SAFE')

            #dbutils.fs.mv("file:/fsdh/workdir/S2A_MSIL2A_20180101T160031_N9999_R054_T18TVQ_20230815T183818.SAFE","/ccrs/batch/data/sen2cor/outputdir/S2A_MSIL2A_20180101T160031_N9999_R054_T18TVQ_20230815T183818.SAFE",recurse=True)

            break

            # uncomment if you want to process all files
            # subprocess.run(['/fsdh/s2/sen2cor021100/bin/L2A_Process','--output_dir',options['wDirName'],iFileName],shell=True,capture_output=True)
            #dbutils.fs.mv("file"+options['wDirName',options['oDirName'],,recurse=True)

