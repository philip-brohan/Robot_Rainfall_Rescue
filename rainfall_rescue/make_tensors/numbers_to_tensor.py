#!/usr/bin/env python

# Convert rainfall-rescued numbers to a target tensor

import os
import sys

import tensorflow as tf
import numpy
import csv

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filen", help="Document name", type=str, required=True)
parser.add_argument("--test", help="test data, not training", action="store_true")
parser.add_argument(
    "--opfile", help="tf data file name", default=None, type=str, required=False
)
args = parser.parse_args()
if args.opfile is None:
    purpose = "training"
    if args.test:
        purpose = "test"
    args.opfile = "%s/Robot_Rainfall_Rescue/from_Ed/tensors/numbers/%s/%s.tfd" % (
        os.getenv("SCRATCH"),
        purpose,
        args.filen[:-4],
    )

# Pad transcribed values so they are in form %5.2f ('ab.cd')
def reformatNumber(mv):
    if len(mv) == 0:
        return "  .  "
    if float(mv) > 99 or float(mv) < 0:
        return "  .  "
    return "%5.2f" % float(mv)


# Load the numbers
numbers = []
with open(
    "%s/Robot_Rainfall_Rescue/from_Ed/csvs/%s" % (os.getenv("SCRATCH"), args.filen)
) as csvf:
    csvr = csv.reader(csvf)
    csvl = list(csvr)
    # Add the year digits
    yr = csvl[4]
    for yi in range(1, 11):
        for di in range(4):
            numbers.append(int(yr[yi][di]))
    # Add the monthly precip
    for mi in range(12):
        for yi in range(1, 11):
            mv = reformatNumber(csvl[5 + mi][yi])
            for di in (0, 1, 3, 4):
                numbers.append(mv[di])


# There are 10*12*4 = 480 digits/image, as well as the years
#  - an additional 10*4 => 520 digits

# Target is a set of 11 probabilities - (' ',0,1,2,...,9) - for each number.
# Each probability is a floating point number on the range 0-1.
# Here they are all 0 or 1, because we know the values exactly, but
#  this provides a suitable target for an ML estimator.
target = numpy.zeros((520, 11))
for idx in range(520):
    if numbers[idx] == " ":
        target[idx, 10] = 1.0
    else:
        target[idx, int(numbers[idx])] = 1.0
    idx += 1

ict = tf.convert_to_tensor(target, numpy.float32)

if not os.path.isdir(os.path.dirname(args.opfile)):
    try:  # These calls sometimes collide
        os.makedirs(os.path.dirname(args.opfile))
    except FileExistsError:
        pass

# Write to file
sict = tf.io.serialize_tensor(ict)
tf.io.write_file(args.opfile, sict)
