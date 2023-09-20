import datetime
import os
import subprocess
import zipfile

import batch_helpers as bh

os.system('mkdir -p /fsdh/workdir')

# sen2cor options
options = {'L1ADirName' : '/ccrs/batch/data/',
           'iDirName' : '/ccrs/batch/data/sen2cor/inputdir/', 
           'wDirName' : '/fsdh/workdir/',
           'oDirName' : '/ccrs/batch/data/sen2cor/outputdir2/'}

# We setup our connection to the batch account, and define the specifications of the pool we want to use, and the job we want to run the tasks in.
timeout = datetime.timedelta(minutes=25)
batch_client = bh.create_batch_client()
pool_info = bh.create_pool_if_not_exist(batch_client, 'BatchPool', 'STANDARD_DS1_V2', 1, mount_storage=True)
job_id = bh.create_job_if_not_exist(batch_client, 'test', pool_info)

# We create a list of task ids to monitor.
task_ids = []

# We iterate through all necessary items.
for item in os.listdir(options['L1ADirName']):
    if item.endswith('zip'):
        with zipfile.ZipFile(options['L1ADirName']+item, 'r') as zip_ref:

            #  We use the batch helper to execute the bash script as a batch task under the job we created above.
            #  We can easily pass in arguments to the bash script as a list of strings.
            task_id = bh.execute_batch_script('/fsdh/s2/sen2cor021100/bin/L2A_Process', job_id, 'sen2cor',  [f"--output_dir {options['oDirName']}", f"--input_dir {options['iDirName']}"])
            task_ids.append(task_id)

# We monitor the status of all tasks under our job.
bh.wait_for_tasks_to_complete(batch_client, job_id, timeout)

# We aggregate the outputs and exceptions of given task list under our job.
[out, exc] = bh.print_task_output(batch_client, job_id, task_ids)

# We clean up the pool, job, and tasks we created.
bh.clean_up(batch_client, pool_info.pool_id, job_id, task_ids)