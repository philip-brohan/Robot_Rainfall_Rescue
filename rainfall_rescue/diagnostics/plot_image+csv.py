#!/usr/bin/env python

# Plot a 10-year monthly rainfall image and the data Rainfall Rescue got from it

from rainfall_rescue.utils.pairs import get_index_list, load_pair
import random
import PIL.Image
import json
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--label",
    help="Image identifier",
    type=str,
    required=False,
    default=None,
)
args = parser.parse_args()
if args.label is None:
    args.label = random.choice(get_index_list())


# load the imagedata pair
img, csv = load_pair(args.label)

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
ax_original.set_axis_off()
imgplot = ax_original.imshow(img, zorder=10)

# Metadata top right
ax_metadata = fig.add_axes([0.52, 0.8, 0.47, 0.15])
ax_metadata.set_xlim(0, 1)
ax_metadata.set_ylim(0, 1)
ax_metadata.set_xticks([])
ax_metadata.set_yticks([])
ax_metadata.text(
    0.05,
    0.8,
    "Station Number: %s" % csv["Number"],
    fontsize=12,
    color="black",
)
ax_metadata.text(
    0.05,
    0.7,
    "Location: %s" % csv["Name"],
    fontsize=12,
    color="black",
)

years = [int(x) for x in csv["Years"]]
years = sorted(years)

# Digitised numbers on the right
ax_digitised = fig.add_axes([0.52, 0.13, 0.47, 0.63])
ax_digitised.set_xlim(years[0] - 0.5, years[-1] + 0.5)
ax_digitised.set_xticks(range(years[0], years[-1] + 1))
ax_digitised.set_xticklabels(years)
ax_digitised.set_ylim(0.5, 12.5)
ax_digitised.set_yticks(range(1, 13))
ax_digitised.set_yticklabels(
    (
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    )
)
ax_digitised.xaxis.set_ticks_position("top")
ax_digitised.xaxis.set_label_position("top")
ax_digitised.invert_yaxis()
ax_digitised.set_aspect("auto")

monthNumbers = {
    "Jan": 1,
    "January": 1,
    "Feb": 2,
    "February": 2,
    "Mar": 3,
    "March": 3,
    "Apr": 4,
    "April": 4,
    "May": 5,
    "Jun": 6,
    "June": 6,
    "Jul": 7,
    "July": 7,
    "Aug": 8,
    "August": 8,
    "Sep": 9,
    "September": 9,
    "Oct": 10,
    "October": 10,
    "Nov": 11,
    "November": 11,
    "Dec": 12,
    "December": 12,
}
for year in years:
    for month in monthNumbers.keys():
        try:
            ax_digitised.text(
                year,
                monthNumbers[month],
                csv[month][year - min(years)],
                ha="center",
                va="center",
                fontsize=12,
                color="black",
            )
        except KeyError as e:
            continue

# Totals along the bottom
ax_totals = fig.add_axes([0.52, 0.09, 0.47, 0.03])
ax_totals.set_xlim(0.5, 10.5)
ax_totals.set_ylim(0, 1)
ax_totals.set_xticks([])
ax_totals.set_yticks([])

for i, total in enumerate(csv["Totals"]):
    ax_totals.text(
        i + 1,
        0.5,
        total,
        ha="center",
        va="center",
        fontsize=12,
        color="black",
    )

# Render
fig.savefig(
    "%s.webp" % (args.label),
)
