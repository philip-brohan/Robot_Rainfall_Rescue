#!/usr/bin/env python

# Convert all the fake rainfall images to tensors for ML model training

# This script does not run the commands - it makes a list of commands
#  (in the file 'run.txt') which can be run in parallel.

import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--subdir", help="Dataset sub-directory", type=str, required=True)
args = parser.parse_args()

rootd = "%s/Robot_Rainfall_Rescue/training_data/cell/%s" % (
    os.getenv("SCRATCH"),
    args.subdir,
)


f = open("run_i2t.sh", "w+")

for doci in range(100000):
    if os.path.isfile("%s/tensors/images/%05d.tfd" % (rootd, doci)):
        continue
    cmd = ('./image_to_tensor.py --rootd=%s --docn="%05d"\n') % (rootd, doci)
    f.write(cmd)

f.close()
