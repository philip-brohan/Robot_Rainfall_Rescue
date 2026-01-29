#!/usr/bin/env python

# Plot a Daily Rainfall image along with consensus data from all models for comparison
import sys

from daily_rainfall.utils.load import (
    load_image,
    load_json,
    image_id_to_filename,
    image_id_to_transcription_filename,
)
from daily_rainfall.utils.validate import (
    plot_image,
    plot_daily_table_consensus,
    plot_totals_consensus,
)

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", action="append", default=[], help="Names of models to use"
)
parser.add_argument(
    "--image",
    help="Image identifier",
    type=str,
    required=True,
)
args = parser.parse_args()
if len(args.model) < 2:
    raise ValueError("Please provide at least two model IDs using --model")

# load the image
img = load_image(image_id_to_filename(args.image))

# Load the model extracted data
extracted = []
for model_id in args.model:
    fname = image_id_to_transcription_filename(args.image, group=model_id)
    try:
        data = load_json(fname)
        extracted.append(data)
    except Exception:
        print(f"Failed load of {fname}")
        sys.exit(1)


# Create the figure
fig = Figure(
    figsize=(13, 10),  # Width, Height (inches)
    dpi=100,
    facecolor=(0.95, 0.95, 0.95, 1),
    edgecolor=None,
    linewidth=0.0,
    frameon=True,
    subplotpars=None,
    tight_layout=None,
)
canvas = FigureCanvas(fig)

# Image in the left
ax_original = fig.add_axes([0.01, 0.02, 0.47, 0.96])
plot_image(ax_original, img)

# First model in the middle
ax_digitised1 = fig.add_axes([0.52, 0.13, 0.47, 0.63])
plot_daily_table_consensus(ax_digitised1, extracted)
ax_totals1 = fig.add_axes([0.52, 0.05, 0.47, 0.07])
plot_totals_consensus(ax_totals1, extracted)


# Render
fig.savefig("consensus.webp")
