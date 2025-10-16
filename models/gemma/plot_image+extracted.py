#!/usr/bin/env python

# Plot a 10-year monthly rainfall image and the data Gemma got from it

from rainfall_rescue.utils.pairs import get_index_list, load_pair, csv_to_json
from rainfall_rescue.utils.validate import (
    load_extracted,
    plot_image,
    plot_metadata,
    plot_monthly_table,
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
    default="google/gemma-3-4b-it",
)
parser.add_argument(
    "--label",
    help="Image identifier",
    type=str,
    required=False,
    default=None,
)
parser.add_argument(
    "--fake",
    help="Use fake data - not real",
    action="store_true",
    required=False,
    default=False,
)
args = parser.parse_args()
if args.label is None:
    args.label = random.choice(get_index_list(fake=args.fake))

if len(args.label) < 5:
    args.fake = True


# load the image/data pair
img, csv = load_pair(args.label)
jcsv = json.loads(csv_to_json(csv))


# Load the model extracted data
extracted = load_extracted(args.model_id, args.label)

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

# Metadata top right
ax_metadata = fig.add_axes([0.52, 0.8, 0.47, 0.15])
plot_metadata(ax_metadata, extracted, jcsv)


# Digitised numbers on the right
ax_digitised = fig.add_axes([0.52, 0.13, 0.47, 0.63])
plot_monthly_table(ax_digitised, extracted, jcsv)

# Totals along the bottom
ax_totals = fig.add_axes([0.52, 0.05, 0.47, 0.07])
plot_totals(ax_totals, extracted, jcsv)

# Render
fig.savefig("extracted.webp")
