#!/usr/bin/env python

# Show sub-images in a random sample of RR images

import os
import sys

import cv2
import numpy as np
import random


import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

sys.path.append("%s/../../image_filtering/openCV_polygons" % os.path.dirname(__file__))
from optimise_xy import imageToBW
from optimise_xy import imageToLines
from optimise_xy import overplotVPoints
from optimise_xy import fitVlines
from optimise_xy import findVOS
from optimise_xy import overplotVFit
from optimise_xy import overplotHPoints
from optimise_xy import fitHlines
from optimise_xy import findHOS
from optimise_xy import overplotHFit

# 5x2 grid of images
fig = Figure(
    figsize=(5 * 10.24, 2 * 10.24),
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

# Extract and image region with 1 month's data
def getSubImage(sImage, yIdx, mIdx, vfit, hfit):
    pad = 5
    pLeft = int(vfit[0] + vfit[1] * (yIdx - 1) - pad)
    pRight = int(vfit[0] + vfit[1] * yIdx + pad)
    hSpace = hfit[2] / 13
    pTop = int(hfit[0] - hfit[1] - hfit[2] + hSpace * (mIdx - 0.5) - pad)
    pBottom = int(hfit[0] - hfit[1] - hfit[2] + hSpace * (mIdx + 0.5) + pad)
    subImage = sImage[pTop:pBottom, pLeft:pRight].copy()
    return subImage


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
    # ax_original.set_xlim([0, 1024])
    # ax_original.set_ylim([1632, 0])
    ax_original.set_aspect("auto")

    vfit = findVOS(pLines["vertical"])
    hfit = findHOS(pLines["horizontal"], pLines["vertical"], vfit)

    for yIdx in range(1, 11):
        for mIdx in range(1, 13):
            subImg = getSubImage(sImage, yIdx, mIdx, vfit, hfit)
            ax_sub = ax_original.inset_axes(
                [yIdx / 11 - 0.45 / 11, 1 - mIdx / 13 - 0.45 / 13, 0.9 / 11, 0.9 / 13]
            )
            ax_sub.set_axis_off()
            ax_sub.set_aspect("equal")
            ax_sub.imshow(cv2.cvtColor(subImg, cv2.COLOR_BGR2RGB))


# Show a set of images
imNames = os.listdir("%s/Robot_Rainfall_Rescue/from_Ed/images" % os.getenv("SCRATCH"))
for i in range(10):
    rImage = random.choice(imNames)
    print(rImage)
    showImage(rImage, spi=i + 1)

# Render the figure as a png
fig.savefig("multi_split.png")
