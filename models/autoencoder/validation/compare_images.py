#!/usr/bin/env python

# Compare one of the test images - original v. autoencoded

import os
import sys

import tensorflow as tf
import numpy
import itertools

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

sys.path.append("%s/../" % os.path.dirname(__file__))
from autoencoderModel import autoencoderModel
from makeDataset import getImageDataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=25)
parser.add_argument(
    "--image", help="Test image number", type=int, required=False, default=0
)
args = parser.parse_args()

# Set up the model and load the weights at the chosen epoch
autoencoder = autoencoderModel()
weights_dir = ("%s/ML_ten_year_rainfall/autoencoder/" + "Epoch_%04d") % (
    os.getenv("SCRATCH"),
    args.epoch - 1,
)
load_status = autoencoder.load_weights("%s/ckpt" % weights_dir)
# Check the load worked
load_status.assert_existing_objects_matched()

# Get test case number args.image
testData = getImageDataset(purpose="test", nImages=args.image + 1)
testData = testData.batch(1)
original = next(itertools.islice(testData, args.image, args.image + 1))

# Run that test case through the autoencoder
encoded = autoencoder.predict_on_batch(original)

# Plot original and encoded side-by-side as images
fig = Figure(
    figsize=(16.28, 11.25),
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
ax_full.add_patch(
    matplotlib.patches.Rectangle((0, 0), 1, 1, fill=True, facecolor="white")
)

# Original
ax_original = fig.add_axes([0.02, 0.015, 0.47, 0.97])
ax_original.set_axis_off()
ax_original.matshow(tf.reshape(original, [1024, 768, 3]))

# Encoded
ax_encoded = fig.add_axes([0.51, 0.015, 0.47, 0.97])
ax_encoded.set_axis_off()
ax_encoded.matshow(tf.reshape(encoded, [1024, 768, 3]))

# Render the figure as a png
fig.savefig("compare.png")
