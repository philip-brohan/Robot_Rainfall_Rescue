#!/usr/bin/env python

# Plot a 10-year monthly rainfall image and test to see
# how well multiple models agree on the digitised values.

from rainfall_rescue.utils.pairs import get_index_list, load_pair, csv_to_json
from rainfall_rescue.utils.validate import (
    load_extracted,
    plot_image,
    plot_metadata_agreement,
    plot_monthly_table_agreement,
    plot_totals_agreement,
)
import random
import os
import json
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_ids",
    help="Model IDs (comma-separated)",
    type=str,
    required=False,
    default="google/gemma-3-4b-it,google/gemma-3-12b-it",
)
parser.add_argument(
    "--agreement_count",
    help="Min. number of models that must agree",
    type=int,
    required=False,
    default=2,
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

# Assemble list of model IDs
model_ids = args.model_ids.split(",")
if len(model_ids) < 2:
    raise ValueError("At least two model IDs are required for agreement plotting.")

# load the image/data pair
img, csv = load_pair(args.label)
jcsv = json.loads(csv_to_json(csv))

# Load the model extracted data
extracted = {}
for model_id in model_ids:
    extracted[model_id] = load_extracted(model_id, args.label)


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
ax_metadata1 = fig.add_axes([0.52, 0.8, 0.47, 0.15])
plot_metadata_agreement(
    ax_metadata1, extracted, jcsv, agreement_count=args.agreement_count
)

ax_digitised1 = fig.add_axes([0.52, 0.13, 0.47, 0.63])
plot_monthly_table_agreement(
    ax_digitised1, extracted, jcsv, agreement_count=args.agreement_count
)

ax_totals1 = fig.add_axes([0.52, 0.05, 0.47, 0.07])
plot_totals_agreement(ax_totals1, extracted, jcsv, agreement_count=args.agreement_count)


# Render
fig.savefig("agree.webp")
