#!/usr/bin/env python

# Plot a Daily Rainfall image along with extracted data from two different models for comparison

from daily_rainfall.utils.load import (
    load_image,
    load_json,
    image_id_to_filename,
    image_id_to_transcription_filename,
)
from daily_rainfall.utils.validate import (
    plot_image,
    plot_daily_table,
    plot_totals,
)

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id_1",
    help="Model ID",
    type=str,
    required=False,
    default="training",
)
parser.add_argument(
    "--model_id_2",
    help="Model ID",
    type=str,
    required=False,
    default="HuggingFaceTB/SmolVLM-Instruct",
)
parser.add_argument(
    "--image",
    help="Image identifier",
    type=str,
    required=True,
)
args = parser.parse_args()

# load the image
img = load_image(image_id_to_filename(args.image))

# Load the model extracted data
extracted_1 = load_json(
    image_id_to_transcription_filename(args.image, group=args.model_id_1)
)
extracted_2 = load_json(
    image_id_to_transcription_filename(args.image, group=args.model_id_2)
)

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

# Image in the left
ax_original = fig.add_axes([0.01, 0.02, 0.32, 0.96])
plot_image(ax_original, img)

# First model in the middle
ax_digitised1 = fig.add_axes([0.34, 0.13, 0.31, 0.63])
plot_daily_table(ax_digitised1, extracted_1, group=args.model_id_1)
ax_totals1 = fig.add_axes([0.34, 0.05, 0.31, 0.07])
plot_totals(ax_totals1, extracted_1, group=args.model_id_1)


# Second model on the right
ax_digitised2 = fig.add_axes([0.68, 0.13, 0.31, 0.63])
plot_daily_table(
    ax_digitised2, extracted_2, group=args.model_id_2, comparison=extracted_1
)
ax_totals2 = fig.add_axes([0.68, 0.05, 0.31, 0.07])
plot_totals(ax_totals2, extracted_2, group=args.model_id_2, comparison=extracted_1)

# Render
fig.savefig("compare.webp")
