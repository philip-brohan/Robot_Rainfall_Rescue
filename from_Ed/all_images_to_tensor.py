#!/usr/bin/env python

# Convert all the Ten-year rainfall images to tensors for ML model training

# Partition off 1/10 of them to be test data

# This script does not run the commands - it makes a list of commands
#  (in the file 'run.txt') which can be run in parallel.

import os

rootd = "%s/ML_ten_year_rainfall/" % os.getenv("SCRATCH")

# Function to check if the job is already done for this timepoint
def is_done(filen, purpose):
    op_file_name = "%s/tensors/images/%s/%s.tfd" % (rootd, purpose, filen[:-4],)
    if os.path.isfile(op_file_name):
        return True
    return False


f = open("run_i2t.sh", "w+")

count = 1
files = sorted(os.listdir("%s/from_Ed/images" % rootd))
for filen in files:
    if count % 10 == 0:
        if not is_done(filen, "test"):
            cmd = ('./image_to_tensor.py --filen="%s" --test\n') % filen
            f.write(cmd)
    else:
        if not is_done(filen, "training"):
            cmd = ('./image_to_tensor.py --filen="%s"\n') % filen
            f.write(cmd)
    count += 1

f.close()
