#!/usr/bin/env python

# Convert all the fake rainfall images to tensors for ML model training

# This script does not run the commands - it makes a list of commands
#  (in the file 'run.txt') which can be run in parallel.

import os

rootd = "%s/ML_ten_year_rainfall/fakes/plain/images/" % os.getenv("SCRATCH")

# Function to check if the job is already done for this timepoint
def is_done(dirn):
    op_file_name = "%s%s.tfd" % (os.getenv("SCRATCH"), dirn)
    if os.path.isfile(op_file_name):
        return True
    return False


f = open("run_i2t.sh", "w+")

for doci in range(10000):
    cmd = (
        "conda activate ml_ten_year_rainfall; " + './image_to_tensor.py --docn="%04d"\n'
    ) % doci
    f.write(cmd)

f.close()
