#!/usr/bin/env python

# Convert a double-page logbook image to a 1024px-square tensor

import os
import sys

import tensorflow as tf
import numpy
from PIL import Image

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dirn", help="Directory name", type=str, required=True)
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
    args.opfile = ("%s/ML_logbooks/images/" + "%s/%s/%s.tfd") % (
        os.getenv("SCRATCH"),
        purpose,
        args.dirn,
        args.filen,
    )

# Load the image as data
image = Image.open(
    "%s/logbook_images/NA_WW1/%s/%s" % (os.getenv("SCRATCH"), args.dirn, args.filen)
)

# Basic check to see if it's a double page spread - give up otherwise
(width, height) = image.size
if width < 4000 or width < height:
    sys.exit(1)

# Output the tensor

if not os.path.isdir(os.path.dirname(args.opfile)):
    try:  # These calls sometimes collide
        os.makedirs(os.path.dirname(args.opfile))
    except FileExistsError:
        pass

# Convert to Tensor - normalised and scaled
image = image.convert("RGB")
image = image.resize((1024, 1024))
image_n = numpy.array(image, numpy.float32) / 256
ict = tf.convert_to_tensor(image_n, numpy.float32)

# Write to file
sict = tf.io.serialize_tensor(ict)
tf.io.write_file(args.opfile, sict)
