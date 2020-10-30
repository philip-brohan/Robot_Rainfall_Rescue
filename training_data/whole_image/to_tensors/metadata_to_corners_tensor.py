#!/usr/bin/env python

# Make a tensor containing grid corner locations from the image metadata

import os
import sys
import math

import tensorflow as tf
import numpy
import pickle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--docn", help="Document name", type=str, required=True)
args = parser.parse_args()

# Load the metadata
with open(
    "%s/ML_ten_year_rainfall/training_data/meta/%s.pkl"
    % (os.getenv("SCRATCH"), args.docn),
    "rb",
) as pkf:
    mdata = pickle.load(pkf)

# mdata is a dictionary - convert it to a class so contents are attributes
#  and we can share code with tyrImage.
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


mdata = AttrDict(mdata)

# From the metadata, find the corners of the data grid
#  (Eight floats on the range 0-1)
# Functions copied from the ATB2 image class - should reuse that class instead

# Rotate by angle degrees clockwise
def gRotate(self, point, angle=None, origin=None):
    if angle is None:
        angle = self.rotate
    if angle == 0:
        return point
    if origin is None:
        origin = gCentre(self)
    ox, oy = origin[0] * self.pageWidth, origin[1] * self.pageHeight
    px, py = point[0] * self.pageWidth, point[1] * self.pageHeight
    angle = math.radians(angle) * -1
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx / self.pageWidth, qy / self.pageHeight


def gCentre(self):
    return (
        0.5 + self.xshift / self.pageWidth + (self.xscale - 1) * 0.43,
        0.525 + self.yshift / self.pageHeight - (self.yscale - 1) * 0.2,
    )


# Corners of grid
def topLeft(self):
    return (
        0.1 + self.xshift / self.pageWidth,
        0.725 + self.yshift / self.pageHeight,
    )


def topRight(self):
    return (
        0.96 + self.xshift / self.pageWidth + (self.xscale - 1) * 0.86,
        0.725 + self.yshift / self.pageHeight,
    )


def bottomLeft(self):
    return (
        0.1 + self.xshift / self.pageWidth,
        0.325 + self.yshift / self.pageHeight - (self.yscale - 1) * 0.4,
    )


def bottomRight(self):
    return (
        0.96 + self.xshift / self.pageWidth + (self.xscale - 1) * 0.86,
        0.325 + self.yshift / self.pageHeight - (self.yscale - 1) * 0.4,
    )


target = []
target.extend(gRotate(mdata, topLeft(mdata)))
target.extend(gRotate(mdata, topRight(mdata)))
target.extend(gRotate(mdata, bottomLeft(mdata)))
target.extend(gRotate(mdata, bottomRight(mdata)))
ict = tf.convert_to_tensor(target, numpy.float32)

# Output the tensor
opdir = "%s/ML_ten_year_rainfall/training_data/tensors/corners/" % os.getenv("SCRATCH")
if not os.path.isdir(opdir):
    try:  # These calls sometimes collide
        os.makedirs(opdir)
    except FileExistsError:
        pass

# Write to file
sict = tf.io.serialize_tensor(ict)
tf.io.write_file("%s/%s.tfd" % (opdir, args.docn), sict)
