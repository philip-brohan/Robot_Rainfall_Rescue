#!/usr/bin/env python

# Split an image into smaller images each containing one rainfall number

import os
import sys

import tensorflow as tf
import numpy
import itertools
from PIL import Image

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

sys.path.append("%s/../" % os.path.dirname(__file__))
from cornerModel import dcCNR
from makeDataset import getImageDataset
from makeDataset import getOriginalDataset
from makeDataset import getCornersDataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=25)
parser.add_argument(
    "--image", help="Test image number", type=int, required=False, default=0
)
args = parser.parse_args()

# Set up the model and load the weights at the chosen epoch
seeker = dcCNR()
weights_dir = ("%s/Robot_Rainfall_Rescue/models/dcCNR/Epoch_%04d") % (
    os.getenv("SCRATCH"),
    args.epoch - 1,
)
load_status = seeker.load_weights("%s/ckpt" % weights_dir)
# Check the load worked
load_status.assert_existing_objects_matched()

# Get test case number args.image
testImage = getImageDataset(subdir="perturbed",purpose="test", nImages=args.image + 1)
testImage = testImage.batch(1)
originalImage = next(itertools.islice(testImage, args.image, args.image + 1))
testNumbers = getCornersDataset(subdir="perturbed",purpose="test", nImages=args.image + 1)
testNumbers = testNumbers.batch(1)
original = next(itertools.islice(testNumbers, args.image, args.image + 1))
testOI = getOriginalDataset(subdir="perturbed",purpose="test", nImages=args.image + 1)
testOI = testOI.batch(1)
oImage = next(itertools.islice(testOI, args.image, args.image + 1))

# Run that test image through the transcriber
encoded = seeker.predict_on_batch(originalImage)

# Get coordinates for a single rainfall value from the estimated corners
def getCorners(corners,year,month):    
    x = (year+0.5)/10
    y = (month+1)/13
    return(corners[0]*(1-x)*(1-y)+
           corners[2]*(1-x)*y+
           corners[4]*x*(1-y)+
           corners[6]*x*y,
           corners[1]*(1-x)*(1-y)+
           corners[3]*(1-x)*y+
           corners[5]*x*(1-y)+
           corners[7]*x*y)

# Load the original image
def getImage(subdir="perturbed",purpose="test",nimage=0):
    dirBase="%s/Robot_Rainfall_Rescue/training_data/%s/" % (
            os.getenv("SCRATCH"),
            subdir,
    )
    inFiles = os.listdir("%s/%s" % (dirBase, 'images'))
    splitI = int(len(inFiles) * 0.9)
    if purpose == "training":
        inFiles = inFiles[:splitI]
    if purpose == "test":
        inFiles = inFiles[splitI:]
    image = Image.open("%s/images/%s" % (dirBase,inFiles[nimage]))
    return(image)

image = getImage(subdir="perturbed",purpose="test",nimage=args.image-1)
image = image.convert("RGB")
image = image.resize((640, 1024))
image = numpy.array(image, numpy.float32) / 256
image = numpy.mean(image, axis=2)

# Plot original image on the left and the sub-images on the right
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
# Paint the background white - why is this needed?
ax_full = fig.add_axes([0, 0, 1, 1])
ax_full.set_xlim([0, 1])
ax_full.set_ylim([0, 1])
ax_full.set_aspect("auto")
ax_full.add_patch(
    matplotlib.patches.Rectangle((0, 0), 1, 1, fill=True, facecolor="grey")
)
fc = ("red", "blue", "orange", "black")

# Original
ax_original = fig.add_axes([0.02, 0.015, 0.47, 0.97])
ax_original.set_axis_off()
ax_original.set_xlim([0, 640])
ax_original.set_ylim([1024, 0])
ax_original.set_aspect("auto")
ax_original.imshow(
    tf.reshape(oImage, [1024, 640]),
    cmap="gray",
    vmin=0,
    vmax=1,
    #origin="upper",
    #interpolation="nearest",
)
cset = original[0, :]
for year in range(10):
    for month in range(12):
        cpt = getCorners(cset,year,month)
        ax_original.add_patch(
            matplotlib.patches.Circle(
                (cpt[0]*640,(1.0-cpt[1])*1024),
                radius=5,
                facecolor="red",
                edgecolor="red",
                linewidth=0.1,
                alpha=0.8,
            )
        )

# Plot encoded using same method as original plot
ax_encoded = fig.add_axes([0.51, 0.015, 0.47, 0.97])
ax_encoded.set_axis_off()
ax_encoded.set_xlim([0, 640])
ax_encoded.set_ylim([1024, 0])
ax_encoded.imshow(
    tf.reshape(oImage, [1024, 640]),
    cmap="gray",
    vmin=0,
    vmax=1,
)

cset = encoded[0, :]
for year in range(10):
    for month in range(12):
        cpt = getCorners(cset,year,month)
        ax_encoded.add_patch(
            matplotlib.patches.Circle(
                (cpt[0]*640,(1.0-cpt[1])*1024),
                radius=5,
                facecolor="red",
                edgecolor="red",
                linewidth=0.1,
                alpha=0.8,
            )
        )

# Render the figure as a png
fig.savefig("centres.png")
