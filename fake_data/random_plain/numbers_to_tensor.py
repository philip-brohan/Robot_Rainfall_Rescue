#!/usr/bin/env python

# Convert fake rainfall numbers to a target tensor

import os
import sys

import tensorflow as tf
import numpy
import pickle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--docn", help="Document name", type=str, required=True)
args = parser.parse_args()

# Load the numbers
with open(
    "%s/ML_ten_year_rainfall/fakes/plain/numbers/%s.pkl"
    % (os.getenv("SCRATCH"), args.docn),
    "rb",
) as pkf:
    mdata = pickle.load(pkf)


# There are 10*12*3 = 360 digits/image, but we also want the monthly means
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
    inr = 0.0
    for yri in range(10):
        inr += mdata[yri][mni][0] + mdata[yri][mni][1] / 10 + mdata[yri][mni][2] / 100
    inr /= 10
    inr = int(inr * 100)
    for dgt in [int(x) for x in "%03d" % inr]:
        target[idx, dgt] = 1.0
        idx += 1
# Add the annual totals
for yri in range(10):
    inr = 0.0
    for mni in range(12):
        inr += mdata[yri][mni][0] + mdata[yri][mni][1] / 10 + mdata[yri][mni][2] / 100
    inr = int(inr * 100)
    for dgt in [int(x) for x in "%04d" % inr]:
        target[idx, dgt] = 1.0
        idx += 1

ict = tf.convert_to_tensor(target, numpy.float32)

# Output the tensor
opdir = "%s/ML_ten_year_rainfall/fakes/plain/tensors/numbers/" % os.getenv("SCRATCH")
if not os.path.isdir(opdir):
    try:  # These calls sometimes collide
        os.makedirs(opdir)
    except FileExistsError:
        pass

# Write to file
sict = tf.io.serialize_tensor(ict)
tf.io.write_file("%s/%s.tfd" % (opdir, args.docn), sict)
