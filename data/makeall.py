#!/usr/bin/env python

# Convert all the WW1 logbook images to tensors for ML model training

# Partition off 1/10 of them to be test data

# This script does not run the commands - it makes a list of commands
#  (in the file 'run.txt') which can be run in parallel.

import os

# Function to check if the job is already done for this timepoint
def is_done(dirn, filen, purpose):
    op_file_name = "%s/ML_logbooks/images/%s/%s/%s.tfd" % (
        os.getenv("SCRATCH"),
        purpose,
        dirn,
        filen,
    )
    if os.path.isfile(op_file_name):
        return True
    return False


f = open("run.sh", "w+")

dirs = os.listdir("%s/logbook_images/NA_WW1" % os.getenv("SCRATCH"))

count = 1
for dir in dirs:
    files = os.listdir("%s/logbook_images/NA_WW1/%s" % (os.getenv("SCRATCH"), dir))
    for file in files:
        if count % 10 == 0:
            if not is_done(dir, file, "test"):
                cmd = './image_to_tensor.py --dirn="%s" --filen="%s" --test\n' % (
                    dir,
                    file,
                )
                f.write(cmd)
        else:
            if not is_done(dir, file, "training"):
                cmd = './image_to_tensor.py --dirn="%s" --filen="%s"\n' % (dir, file)
                f.write(cmd)
        count += 1

f.close()
