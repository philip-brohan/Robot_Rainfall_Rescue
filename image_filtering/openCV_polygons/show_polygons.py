#!/usr/bin/env python

# Infer polygons from a single-page rainfall image

import os
import sys

import cv2
import numpy as np
import random

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

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

# To greyscale
pImage = cv2.cvtColor(sImage.copy(), cv2.COLOR_BGR2GRAY)
# Gaussian blur to denoise
pImage = cv2.GaussianBlur(pImage, (9, 9), 0)
# Adaptive threshold to convert to BW
pImage = cv2.adaptiveThreshold(
    pImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)
pImage = cv2.bitwise_not(pImage)
kernel = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], np.uint8)
pImage = cv2.dilate(pImage, kernel)


# Find line segments in the processed image
fld = cv2.ximgproc.createFastLineDetector(25, 0.141, 50, 50, 0, True)
lines = fld.detect(pImage.copy())

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


# Overplot the line segments
lcm = matplotlib.cm.get_cmap("hsv", len(lines))
lro = random.sample(list(range(len(lines))), len(lines))
for lsi in range(len(lines)):
    ls = lines[lsi]
    ax_processed.add_line(
        Line2D(
            [ls[0][0], ls[0][2]],
            [ls[0][1], ls[0][3]],
            linewidth=3,
            color=lcm(lro[lsi]),
            zorder=200,
        )
    )


# Render the figure as a png
fig.savefig("processed.png")
