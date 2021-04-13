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

# Plot source and processed images side by side
fig = Figure(
    figsize=(5 * 10.24, 2 * 16.32),
    dpi=100,
    facecolor="white",
    edgecolor="black",
    linewidth=0.0,
    frameon=False,
    subplotpars=None,
    tight_layout=None,
)
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, hspace=0.0, wspace=0.0)
canvas = FigureCanvas(fig)
ax_full = fig.add_axes([0, 0, 1, 1], facecolor="white")


def showImage(filen, spi=1):

    # Load the source image
    fName = "%s/Robot_Rainfall_Rescue/from_Ed/images/%s" % (
        os.getenv("SCRATCH"),
        filen,
    )
    sImage = cv2.imread(fName, cv2.IMREAD_UNCHANGED)
    if sImage is None:
        raise Exception("No such image file %s" % fName)

    # Standardise the size
    sImage = cv2.resize(sImage, (1024, 1632))

    pImage = imageToBW(sImage)

    pLines = imageToLines(pImage)

    ax_original = fig.add_subplot(2, 5, spi)
    ax_original.set_axis_off()
    ax_original.set_xlim([0, 1024])
    ax_original.set_ylim([1632, 0])
    ax_original.set_aspect("auto")
    ax_original.imshow(
        sImage,
    )

    overplotLines(ax_original, pLines)


# Show a set of images
imNames = os.listdir("%s/Robot_Rainfall_Rescue/from_Ed/images" % os.getenv("SCRATCH"))
for i in range(10):
    showImage(random.choice(imNames), spi=i + 1)

# Render the figure as a png
fig.savefig("multi_lines.png")
