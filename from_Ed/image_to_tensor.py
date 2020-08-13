#!/usr/bin/env python

# Convert a single-page rainfall image to a 640x1024px tensor

import os
import sys

import tensorflow as tf
import numpy
from PIL import Image

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filen", help="File name", type=str, required=True)
parser.add_argument("--test", help="test data, not training", action="store_true")
parser.add_argument(
    "--opfile", help="tf data file name", default=None, type=str, required=False
)
args = parser.parse_args()
if args.opfile is None:
    purpose = "training"
    if args.test:
        purpose = "test"
    args.opfile = "%s/ML_ten_year_rainfall/tensors/images/%s/%s.tfd" % (
        os.getenv("SCRATCH"),
        purpose,
        args.filen[:-4],
    )

# Load the image as data
image = Image.open(
    "%s/ML_ten_year_rainfall/from_Ed/images/%s" % (os.getenv("SCRATCH"), args.filen)
)

# Output the tensor

if not os.path.isdir(os.path.dirname(args.opfile)):
    try:  # These calls sometimes collide
        os.makedirs(os.path.dirname(args.opfile))
    except FileExistsError:
        pass

# Convert to Tensor - normalised and scaled
image = image.convert("RGB")
image = image.resize((640, 1024))
image_n = numpy.array(image, numpy.float32) / 256
ict = tf.convert_to_tensor(image_n, numpy.float32)

# Write to file
sict = tf.io.serialize_tensor(ict)
tf.io.write_file(args.opfile, sict)
