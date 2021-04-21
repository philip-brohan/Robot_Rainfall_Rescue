#!/usr/bin/env python

# Find vertical grid lines in a random sample of RR images

import os
import sys

import cv2
import numpy as np

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

# Make the directory to put the sub-images in
sIDir = "%s/Robot_Rainfall_Rescue/sub-images/%s" % (
    os.getenv("SCRATCH"),
    args.filen[:-4],
)
if not os.path.isdir(sIDir):
    os.makedirs(sIDir)

# Standardise the size
sImage = cv2.resize(sImage, (1024, 1632))
# Standarise the colour
pImage = imageToBW(sImage)
# Find line segments in the image
pLines = imageToLines(pImage)
# Find the grid outlines
vfit = findVOS(pLines["vertical"])
hfit = findHOS(pLines["horizontal"], pLines["vertical"], vfit)

# Extract and image region with 1 month's data
def getSubImage(yIdx, mIdx):
    pad = 5
    pLeft = int(vfit[0] + vfit[1] * (yIdx - 1) - pad)
    pRight = int(vfit[0] + vfit[1] * yIdx + pad)
    hSpace = hfit[2] / 13
    pTop = int(hfit[0] - hfit[1] - hfit[2] + hSpace * (mIdx - 0.5) - pad)
    pBottom = int(hfit[0] - hfit[1] - hfit[2] + hSpace * (mIdx + 0.5) + pad)
    subImage = sImage[pTop:pBottom, pLeft:pRight].copy()
    return subImage


# Store all the sub-images
for yIdx in range(1, 11):
    for mIdx in range(1, 13):
        subImg = getSubImage(yIdx, mIdx)
        cv2.imwrite("%s/%02d_%02d.jpg" % (sIDir, yIdx, mIdx), subImg)
