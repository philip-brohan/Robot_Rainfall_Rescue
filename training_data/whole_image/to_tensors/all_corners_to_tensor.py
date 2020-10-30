#!/usr/bin/env python

# Convert all the fake rainfall image corner locations to tensors for ML model training

# This script does not run the commands - it makes a list of commands
#  (in the file 'run_c2t.txt') which can be run in parallel.

import os

rootd = "%s/ML_ten_year_rainfall/training_data/meta/" % os.getenv("SCRATCH")


f = open("run_c2t.sh", "w+")

for doci in range(10000):
    if os.path.isfile(
        "%s/ML_ten_year_rainfall/training_data/tensors/corners/%04d.tfd"
        % (os.getenv("SCRATCH"), doci)
    ):
        continue
    cmd = ('./metadata_to_corners_tensor.py --docn="%04d"\n') % doci
    f.write(cmd)

f.close()
