#!/usr/bin/env python

# Plot a Daily Rainfall image along with consensus data from several models for comparison
# Compare the data from a single hold-out model


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
    get_consensus_monthly_averages,
    get_consensus_daily_averages,
    plot_daily_table,
    plot_totals,
)

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from cartopy import crs as ccrs
from cartopy.io import shapereader


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", action="append", default=[], help="Names of models to use for consensus"
)
parser.add_argument(
    "--hold_out_model",
    type=str,
    required=True,
    help="Name of the model to use for hold-out comparison",
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

# Load the consensus model extracted data
extracted = []
for model_id in args.model:
    fname = image_id_to_transcription_filename(args.image, group=model_id)
    try:
        data = load_json(fname)
        extracted.append(data)
    except Exception:
        print(f"Failed load of {fname}")
        sys.exit(1)

# Load the hold-out model extracted data
hold_out_fname = image_id_to_transcription_filename(
    args.image, group=args.hold_out_model
)
try:
    hold_out_data = load_json(hold_out_fname)
except Exception:
    print(f"Failed load of hold-out model data from {hold_out_fname}")
    sys.exit(1)

# Create the figure
fig = Figure(
    figsize=(20, 10),  # Width, Height (inches)
    dpi=100,
    facecolor=(0.95, 0.95, 0.95, 1),
    edgecolor=None,
    linewidth=0.0,
    frameon=True,
    subplotpars=None,
    tight_layout=None,
)
canvas = FigureCanvas(fig)

# Image on the left
ax_original = fig.add_axes([0.01, 0.02, 0.32, 0.96])
plot_image(ax_original, img)

# Consensus in the middle
ax_digitised1 = fig.add_axes([0.35, 0.13, 0.31, 0.63])
plot_daily_table_consensus(ax_digitised1, extracted)
ax_totals1 = fig.add_axes([0.35, 0.05, 0.31, 0.07])
plot_totals_consensus(ax_totals1, extracted)


# Add top-centre title showing the image identifier, inserting newlines after '/'
title = str(args.image).replace("/", "/\n")
try:
    fig.suptitle(title, x=0.5, y=0.95, ha="center", va="top", fontsize="x-large")
except Exception:
    fig.text(0.5, 0.95, title, ha="center", va="top", fontsize="x-large")

# Get the consensus values for comparison with the hold-out model
consensus_monthly_averages = get_consensus_monthly_averages(extracted, k=2)
consensus_daily_averages = get_consensus_daily_averages(extracted, k=2)

# Hold-out model on the right
ax_digitised2 = fig.add_axes([0.68, 0.13, 0.31, 0.63])
plot_daily_table(ax_digitised2, hold_out_data, comparison=consensus_daily_averages)
ax_totals2 = fig.add_axes([0.68, 0.05, 0.31, 0.07])
plot_totals(
    ax_totals2, hold_out_data, comparison={"Totals": consensus_monthly_averages}
)

# Render
fig.savefig("consensus+metadata+hold-out.webp")
