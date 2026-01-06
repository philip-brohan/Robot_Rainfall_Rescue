#!/usr/bin/env python

# Plot a 10-year monthly rainfall image and the data Gemini3 got from it

from daily_rainfall.utils.load import load_image, load_json, get_json_name
from daily_rainfall.utils.validate import (
    plot_image,
    plot_daily_table,
    plot_totals,
)
import random
import os
import re
import json
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    help="Model ID",
    type=str,
    required=False,
    default="Gemini3",
)
parser.add_argument(
    "--image",
    help="Image path",
    type=str,
    required=True,
)
args = parser.parse_args()

image_file = f"{os.getenv('DOCS')}/Daily_Rainfall_UK/jpgs_300dpi/%s.jpg" % args.image
# load the image/data pair
img = load_image(image_file)
transcription = load_json(get_json_name(image_file, group=args.model_id))


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

# Digitised numbers on the right
ax_digitised = fig.add_axes([0.52, 0.13, 0.47, 0.63])
plot_daily_table(ax_digitised, transcription)

# # Totals along the bottom
ax_totals = fig.add_axes([0.52, 0.05, 0.47, 0.07])
plot_totals(ax_totals, transcription)

# Render
fig.savefig("extracted.webp")
