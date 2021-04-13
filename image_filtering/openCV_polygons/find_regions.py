#!/usr/bin/env python

# Find data regions in a single RR image

import os
import sys

import cv2
import numpy as np
import random

import jenkspy

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

sys.path.append("%s/../dataset" % os.path.dirname(__file__))
from filters import imageToBW
from filters import imageToLines
from filters import overplotLines

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filen", help="File name", type=str, required=True)
args = parser.parse_args()

# Load the source image
fName = "%s/Robot_Rainfall_Rescue/from_Ed/images/%s" % (
    os.getenv("SCRATCH"),
    args.filen,
)
sImage = cv2.imread(fName, cv2.IMREAD_UNCHANGED)
if sImage is None:
    raise Exception("No such image file %s" % fName)

# Standardise the size
sImage = cv2.resize(sImage, (1024, 1632))

pImage = imageToBW(sImage)

pLines = imageToLines(pImage)

# Plot source and processed images side by side
fig = Figure(
    figsize=(16.34, 10.56),
    dpi=100,
    facecolor="white",
    edgecolor="black",
    linewidth=0.0,
    frameon=False,
    subplotpars=None,
    tight_layout=None,
)
canvas = FigureCanvas(fig)
ax_full = fig.add_axes([0, 0, 1, 1], facecolor="white")

ax_original = fig.add_axes([0.02, 0.015, 0.47, 0.97])
ax_original.set_axis_off()
ax_original.set_xlim([0, 1024])
ax_original.set_ylim([1632, 0])
ax_original.set_aspect("auto")
ax_original.imshow(
    sImage,
)

ax_processed = fig.add_axes([0.51, 0.015, 0.47, 0.97])
ax_processed.set_axis_off()
ax_processed.set_xlim([0, 1024])
ax_processed.set_ylim([1632, 0])
ax_processed.imshow(cv2.bitwise_not(pImage), cmap="gray", zorder=100)

overplotLines(ax_processed, pLines)

# Render the figure as a png
fig.savefig("region_lines.png")
