#!/usr/bin/env python

# Convert all the fake rainfall image numbers to tensors for ML model training

# This script does not run the commands - it makes a list of commands
#  (in the file 'run_n2t.txt') which can be run in parallel.

import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--subdir", help="Dataset sub-directory", type=str, required=True)
args = parser.parse_args()

rootd = "%s/Robot_Rainfall_Rescue/training_data/%s" % (os.getenv("SCRATCH"),args.subdir)

f = open("run_n2t.sh", "w+")

for doci in range(10000):
    if os.path.isfile(
        "%s/tensors/numbers/%04d.tfd"
        % (rootd, doci)
    ):
        continue
    cmd = ('./numbers_to_tensor.py --rootd=%s --docn="%04d"\n') % (rootd,doci)
    f.write(cmd)

f.close()
