#!/usr/bin/env python

# Compare one of the test images with original and reconstucted grid points overlaid

import os
import sys

import tensorflow as tf
import numpy
import itertools

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

sys.path.append("%s/../" % os.path.dirname(__file__))
from gridModel import gridModel

sys.path.append("%s/../../dataset" % os.path.dirname(__file__))
from makeDataset import getImageDataset
from makeDataset import getGridDataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=25)
parser.add_argument(
    "--image", help="Test image number", type=int, required=False, default=0
)
args = parser.parse_args()

# Set up the model and load the weights at the chosen epoch
seeker = gridModel()
weights_dir = ("%s/ML_ten_year_rainfall/models/find_grid/" + "Epoch_%04d") % (
    os.getenv("SCRATCH"),
    args.epoch - 1,
)
load_status = seeker.load_weights("%s/ckpt" % weights_dir)
# Check the load worked
load_status.assert_existing_objects_matched()

# Get test case number args.image
testImage = getImageDataset(purpose="test", nImages=args.image + 1)
testImage = testImage.batch(1)
originalImage = next(itertools.islice(testImage, args.image, args.image + 1))
testNumbers = getGridDataset(purpose="test", nImages=args.image + 1)
testNumbers = testNumbers.batch(1)
original = next(itertools.islice(testNumbers, args.image, args.image + 1))

# Run that test image through the transcriber
encoded = seeker.predict_on_batch(originalImage)

# Plot original image on the both sides - overlay original corners on left
#  and reconstructed corners on right
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

# Original
ax_original = fig.add_axes([0.02, 0.015, 0.47, 0.97])
ax_original.set_axis_off()
ax_original.set_xlim([0, 640])
ax_original.set_ylim([1024, 0])
ax_original.set_aspect("auto")
ax_original.imshow(
    tf.reshape(originalImage, [1024, 640, 1]),
    aspect="auto",
    origin="upper",
    interpolation="nearest",
)
cset = original[0, :]
for point in range(120):
    ax_original.add_patch(
        matplotlib.patches.Circle(
            (cset[point * 2] * 640, (1 - cset[point * 2 + 1]) * 1024),
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
ax_original.set_xlim([0, 640])
ax_original.set_ylim([1024, 0])
ax_encoded.imshow(
    tf.reshape(originalImage, [1024, 640, 1]),
    aspect="auto",
    origin="upper",
    interpolation="nearest",
)

cset = encoded[0, :]
for point in range(120):
    ax_encoded.add_patch(
        matplotlib.patches.Circle(
            (cset[point * 2] * 640, (1 - cset[point * 2 + 1]) * 1024),
            radius=5,
            facecolor="red",
            edgecolor="red",
            linewidth=0.1,
            alpha=0.8,
        )
    )


# Render the figure as a png
fig.savefig("individual.png")
