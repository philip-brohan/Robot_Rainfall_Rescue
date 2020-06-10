#!/usr/bin/env python

# Convert a fake rainfall image to a tensor

import os
import sys

import tensorflow as tf
import numpy
from PIL import Image

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--docn", help="Document name", type=str, required=True)
args = parser.parse_args()

# Load the image as data
image = Image.open(
    "%s/ML_ten_year_rainfall/fakes/plain/images/%s.png"
    % (os.getenv("SCRATCH"), args.docn)
)

# Output the tensor
opdir = "%s/ML_ten_year_rainfall/fakes/plain/tensors/images/" % os.getenv("SCRATCH")
if not os.path.isdir(opdir):
    try:  # These calls sometimes collide
        os.makedirs(opdir)
    except FileExistsError:
        pass

# Convert to Tensor - normalised and scaled
image = image.convert("RGB")
image = image.resize((768, 1024))
image_n = numpy.array(image, numpy.float32) / 256
ict = tf.convert_to_tensor(image_n, numpy.float32)

# Write to file
sict = tf.io.serialize_tensor(ict)
tf.io.write_file("%s/%s.tfd" % (opdir, args.docn), sict)
