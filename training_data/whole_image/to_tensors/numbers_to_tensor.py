#!/usr/bin/env python

# Convert fake rainfall numbers to a target tensor

import os
import sys

import tensorflow as tf
import numpy
import pickle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rootd", help="root directory", type=str, required=True)
parser.add_argument("--docn", help="Document name", type=str, required=True)
args = parser.parse_args()

# Load the numbers
with open(
    "%s/numbers/%s.pkl" % (args.rootd, args.docn),
    "rb",
) as pkf:
    mdata = pickle.load(pkf)


# There are 10*12*3 = 360 digits/image, as well as the monthly means
#  - an additional 12*3 - and the yearly totals - 10*4 => 436 digits

# Target is a set of 10 probabilities - one for each digit 0-9 - for each number.
# Each probability is a floating point number on the range 0-1.
# Here they are all 0 or 1, because we know the numbers exactly, but
#  this provides a suitable target for an ML estimator.
target = numpy.zeros((436, 10))
idx = 0
for yri in range(10):
    for mni in range(12):
        for ddx in range(3):
            target[idx, mdata[yri][mni][ddx]] = 1.0
            idx += 1
# Add the monthly means
for mni in range(12):
    for ddx in range(3):
        target[idx, mdata[10][mni][ddx]] = 1.0
        idx += 1
# Add the annual totals
for yri in range(10):
    for ddx in range(4):
        target[idx, mdata[11][yri][ddx]] = 1.0
        idx += 1

ict = tf.convert_to_tensor(target, numpy.float32)

# Output the tensor
opdir = "%s/tensors/numbers/" % args.rootd
if not os.path.isdir(opdir):
    try:  # These calls sometimes collide
        os.makedirs(opdir)
    except FileExistsError:
        pass

# Write to file
sict = tf.io.serialize_tensor(ict)
tf.io.write_file("%s/%s.tfd" % (opdir, args.docn), sict)
