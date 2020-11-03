#!/usr/bin/env python

# Success probability by position on page

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

sys.path.append("%s/../../../dataset" % os.path.dirname(__file__))
from makeDataset import getImageDataset
from makeDataset import getNumbersDataset

sys.path.append("%s/../../../../training_data/whole_image/" % os.path.dirname(__file__))
from tyrImage import tyrImage

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch", type=int, required=False, default=25)
# Set nimages to a small number for fast testing
parser.add_argument(
    "--nimages",
    help="No of test cases to look at",
    type=int,
    required=False,
    default=None,
)
args = parser.parse_args()

# Set up the model and load the weights at the chosen epoch
transcriber = transcriberModel()
weights_dir = (
    "%s/ML_ten_year_rainfall/models/transcriber_full_page/training/" + "Epoch_%04d"
) % (os.getenv("SCRATCH"), args.epoch - 1,)
load_status = transcriber.load_weights("%s/ckpt" % weights_dir)
# Check the load worked
load_status.assert_existing_objects_matched()

# Make the probability matrix
testImages = getImageDataset(purpose="test", nImages=args.nimages)
testNumbers = getNumbersDataset(purpose="test", nImages=args.nimages)
testData = tf.data.Dataset.zip((testImages, testNumbers))
count = 0
pmatrix = numpy.zeros((436))
for testCase in testData:
    image = testCase[0]
    orig = testCase[1]
    encoded = transcriber(tf.reshape(image, [1, 1024, 640, 1]), training=False)
    for tidx in range(436):
        originalDigit = numpy.where(orig[tidx, :] == 1.0)[0][0]
        dgProbabilities = encoded[0, tidx, originalDigit]
        pmatrix[tidx] += dgProbabilities
    count += 1
pmatrix /= count

# Plot encoded using same method as original plot
fig = Figure(
    figsize=(7.68, 10.24),
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
    matplotlib.patches.Rectangle(
        (0, 0), 1, 1, fill=True, facecolor=(0.95, 0.95, 0.95, 1)
    )
)

ti = tyrImage()
ti.drawBox(ax_full)
ti.drawGrid(ax_full)
ti.drawFixedText(ax_full)

# Add the transcribed numbers
tidx = 0
for yri in range(10):
    x = ti.monthsWidth + (yri + 0.5) * (1.0 - ti.meansWidth - ti.monthsWidth) / 10
    tp = ti.topAt(x)
    for mni in range(12):
        lft = ti.leftAt(
            1.0
            - ti.yearHeight
            - (mni + 1) * (1.0 - ti.yearHeight - ti.totalsHeight) / (12 + 1)
        )
        for dgi in range(3):
            ax_full.add_patch(
                matplotlib.patches.Rectangle(
                    (tp[0] - 0.015 + dgi * 0.015 - 0.005, lft[1] - 0.01),
                    0.01,
                    0.02,
                    fill=True,
                    facecolor=(0, 0, 0, 0.1),
                )
            )
            ax_full.add_patch(
                matplotlib.patches.Rectangle(
                    (tp[0] - 0.015 + dgi * 0.015 - 0.005, lft[1] - 0.01),
                    0.01,
                    0.02 * pmatrix[tidx],
                    fill=True,
                    facecolor="blue",
                )
            )
            tidx += 1
# Add the monthly means
tp = ti.topAt(1.0 - ti.meansWidth / 2)
for mni in range(12):
    lft = ti.leftAt(
        1.0
        - ti.yearHeight
        - (mni + 1) * (1.0 - ti.yearHeight - ti.totalsHeight) / (12 + 1)
    )
    for dgi in range(3):
        ax_full.add_patch(
            matplotlib.patches.Rectangle(
                (tp[0] - 0.015 + dgi * 0.015 - 0.005, lft[1] - 0.01),
                0.01,
                0.02,
                fill=True,
                facecolor=(0, 0, 0, 0.1),
            )
        )
        ax_full.add_patch(
            matplotlib.patches.Rectangle(
                (tp[0] - 0.015 + dgi * 0.015 - 0.005, lft[1] - 0.01),
                0.01,
                0.02 * pmatrix[tidx],
                fill=True,
                facecolor="blue",
            )
        )
        tidx += 1
# Add the annual totals
lft = ti.leftAt(ti.totalsHeight / 2)
for yri in range(10):
    x = ti.monthsWidth + (yri + 0.5) * (1.0 - ti.meansWidth - ti.monthsWidth) / 10
    tp = ti.topAt(x)
    inr = 0.0
    for dgi in range(4):
        ax_full.add_patch(
            matplotlib.patches.Rectangle(
                (tp[0] - 0.0225 + dgi * 0.015 - 0.005, lft[1] - 0.01),
                0.01,
                0.02,
                fill=True,
                facecolor=(0, 0, 0, 0.1),
            )
        )
        ax_full.add_patch(
            matplotlib.patches.Rectangle(
                (tp[0] - 0.0225 + dgi * 0.015 - 0.005, lft[1] - 0.01),
                0.01,
                0.02 * pmatrix[tidx],
                fill=True,
                facecolor="blue",
            )
        )
        tidx += 1


# Render the figure as a png
fig.savefig("place.png")
