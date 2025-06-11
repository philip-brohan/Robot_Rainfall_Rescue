#!/usr/bin/env python

# Plot a 10-year monthly rainfall image and the data Gemma got from it

from rainfall_rescue.utils.pairs import get_index_list, load_pair
import random
import os
import json
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_id",
    help="Identifier for this training run",
    type=str,
    required=True,
    default="google/gemma-3-4b-it",
)
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


# load the image/data pair
img, csv = load_pair(args.label)

# Load the model extracted data
exfile = f"{os.getenv('SCRATCH')}/Robot_Rainfall_Rescue/extracted/{args.run_id}/{args.label}.json"
with open(exfile, "r") as f:
    extracted = json.loads(f.read())

# print(extracted)

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


years = [int(x) for x in csv["Years"]]
years = sorted(years)

# Metadata top right
ax_metadata = fig.add_axes([0.52, 0.8, 0.47, 0.15])
ax_metadata.set_xlim(0, 1)
ax_metadata.set_ylim(0, 1)
ax_metadata.set_xticks([])
ax_metadata.set_yticks([])
ext = extracted["start_year"]
color = "red"
try:
    if int(ext) == years[0]:  # Start year extracted correctly
        color = "black"
except:
    pass
ax_metadata.text(
    0.05,
    0.2,
    "Start Year: %s" % ext,
    fontsize=12,
    color=color,
)

# Digitised numbers on the right
ax_digitised = fig.add_axes([0.52, 0.13, 0.47, 0.63])
ax_digitised.set_xlim(years[0] - 0.5, years[-1] + 0.5)
ax_digitised.set_xticks(range(years[0], years[-1] + 1))
ax_digitised.set_xticklabels(years)
lcolor = "red"
try:
    if int(extracted["start_year"]) == years[0]:
        lcolor = "black"
except:
    pass
for label in ax_digitised.get_xticklabels():
    label.set_color(lcolor)
ax_digitised.set_ylim(0.5, 13)
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
            exv = extracted["rainfall"]["%s" % (year)][month]
            if exv == "null":
                exv = 0.0
            rrv = csv[month][year - min(years)]
            if rrv == "":
                rrv = 0.0
            if exv == rrv:
                ax_digitised.text(
                    year,
                    monthNumbers[month],
                    exv,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="black",
                )
            else:
                ax_digitised.text(
                    year,
                    monthNumbers[month],
                    exv,  # "%.2f" % exv,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="red",
                )
                ax_digitised.text(
                    year,
                    monthNumbers[month] + 0.5,
                    "%.2f" % float(rrv),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="blue",
                )
        except KeyError as e:
            continue


# Totals along the bottom
ax_totals = fig.add_axes([0.52, 0.06, 0.47, 0.05])
ax_totals.set_xlim(0.5, 10.5)
ax_totals.set_ylim(0, 1)
ax_totals.set_xticks([])
ax_totals.set_yticks([])

ext = extracted["totals"]
cst = csv["Totals"]
for i in range(10):
    try:
        total = ext[i]
        rrv = cst[i]
        if rrv == "":
            rrv = "null"
        if total == rrv:
            ax_totals.text(
                i + 1,
                0.5,
                total,
                ha="center",
                va="center",
                fontsize=12,
                color="black",
            )
        else:
            ax_totals.text(
                i + 1,
                0.75,
                total,
                ha="center",
                va="center",
                fontsize=12,
                color="red",
            )
            ax_totals.text(
                i + 1,
                0.75 - 0.5,
                rrv,
                ha="center",
                va="center",
                fontsize=12,
                color="blue",
            )
    except KeyError as e:
        continue

# Render
opfile = exfile[:-5] + ".webp"
fig.savefig(opfile)
