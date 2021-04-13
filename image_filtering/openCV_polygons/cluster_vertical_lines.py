#!/usr/bin/env python

# Find nearly-vertical lines in a single-page rainfall image

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

# Find the nearly-vertical lines in the centre of the image
nv = []
for line in lines:
    if max(line[0][1], line[0][3]) > 1200:
        continue
    if min(line[0][1], line[0][3]) < 300:
        continue
    if (abs(line[0][0] - line[0][2]) / abs(line[0][1] - line[0][3])) < 0.1:
        nv.append(line)
lines = np.array(nv)

# Cluster the lines in the x direction
xs = []
for line in lines:
    xs.append(line[0][0])
    xs.append(line[0][2])

jb = jenkspy.jenks_breaks(xs, nb_class=13)

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
lcm = matplotlib.cm.get_cmap("hsv", len(jb))
for lsi in range(len(lines)):
    ls = lines[lsi]
    col = None
    for d in range(len(jb)):
        if ls[0][0] > jb[d] and ls[0][0] < jb[d + 1]:
            ax_processed.add_line(
                Line2D(
                    [ls[0][0], ls[0][2]],
                    [ls[0][1], ls[0][3]],
                    linewidth=3,
                    color=lcm(d),
                    zorder=200,
                )
            )
            continue


# Render the figure as a png
fig.savefig("vertical_lines.png")
