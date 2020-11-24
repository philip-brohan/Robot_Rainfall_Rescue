#!/usr/bin/env python

# Compare one of the test images - original v. transcribed

import os
import sys

import tensorflow as tf
import numpy
import itertools

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

sys.path.append("%s/../" % os.path.dirname(__file__))
from transcriberModel import transcriberModel

sys.path.append("%s/../../dataset" % os.path.dirname(__file__))
from makeDataset import getImageDataset
from makeDataset import getNumbersDataset

from tyrImage import tyrImage

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=25)
parser.add_argument(
    "--image", help="Test image number", type=int, required=False, default=0
)
args = parser.parse_args()

# Set up the model and load the weights at the chosen epoch
transcriber = transcriberModel()
weights_dir = ("%s/Robot_Rainfall_Rescue/models/ATB2_retuned/" + "Epoch_%04d") % (
    os.getenv("SCRATCH"),
    args.epoch - 1,
)
load_status = transcriber.load_weights("%s/ckpt" % weights_dir)
# Check the load worked
load_status.assert_existing_objects_matched()

# Get test case number args.image
testImage = getImageDataset(purpose="test", nImages=args.image + 1)
testImage = testImage.batch(1)
originalImage = next(itertools.islice(testImage, args.image, args.image + 1))
testNumbers = getNumbersDataset(purpose="test", nImages=args.image + 1)
testNumbers = testNumbers.batch(1)
originalNumbers = next(itertools.islice(testNumbers, args.image, args.image + 1))

# Run that test image through the transcriber
encoded = transcriber.predict_on_batch(originalImage)

# Plot original image on the left - make an image from the encoded numbers
#  on the right
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
ax_full.add_patch(
    matplotlib.patches.Rectangle((0, 0), 1, 1, fill=True, facecolor="white")
)

# Original
ax_original = fig.add_axes([0.02, 0.015, 0.47, 0.97])
ax_original.set_axis_off()
ax_original.matshow(tf.reshape(originalImage, [1024, 640, 1]))

# Plot encoded using same method as original plot
ax_encoded = fig.add_axes([0.51, 0.015, 0.47, 0.97])
ax_encoded.set_xlim([0, 1])
ax_encoded.set_ylim([0, 1])
ax_encoded.set_axis_off()

ti = tyrImage()
ti.drawBox(ax_encoded)
ti.drawGrid(ax_encoded)
ti.drawFixedText(ax_encoded)
ti.drawNumbers(ax_encoded, originalNumbers, encoded)
ti.drawMeans(ax_encoded, originalNumbers, encoded)
ti.drawTotals(ax_encoded, originalNumbers, encoded)

# Render the figure as a png
fig.savefig("compare.png")
