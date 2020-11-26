#!/usr/bin/env python

# Convert a fake rainfall image to a tensor

import os
import sys

import tensorflow as tf
import numpy
from PIL import Image

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rootd", help="root directory", type=str, required=True)
parser.add_argument("--docn", help="Document name", type=str, required=True)
args = parser.parse_args()

# Load the image as data
image = Image.open(
    "%s/images/%s.png"
    % (args.rootd, args.docn)
)

# Output the tensor
opdir = "%s/tensors/images/" % args.rootd
if not os.path.isdir(opdir):
    try:  # These calls sometimes collide
        os.makedirs(opdir)
    except FileExistsError:
        pass

# Convert to Tensor - normalised and scaled
image = image.convert("RGB")
image = image.resize((640, 1024))
image_n = numpy.array(image, numpy.float32) / 256
# To monochrome - mean of the colour values
image_n = numpy.mean(image_n, axis=2)
# Auto levels - scale to range 0-1
image_n -= numpy.min(image_n)
image_n /= numpy.max(image_n)
ict = tf.convert_to_tensor(image_n, numpy.float32)

# Write to file
sict = tf.io.serialize_tensor(ict)
tf.io.write_file("%s/%s.tfd" % (opdir, args.docn), sict)
