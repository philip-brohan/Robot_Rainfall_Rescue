#!/usr/bin/env python

# Plot a 10-year monthly rainfall image and compare the data
#  two gemma versions got from it.

from rainfall_rescue.utils.pairs import get_index_list, load_pair
import random
import os
import json
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id_1",
    help="Model ID",
    type=str,
    required=False,
    default="google/gemma-3-4b-it",
)
parser.add_argument(
    "--model_id_2",
    help="Model ID",
    type=str,
    required=False,
    default="google/gemma-3-12b-it",
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
opfile = f"{os.getenv('PDIR')}/extracted/{args.model_id_1}/{args.label}.json"
with open(opfile, "r") as f:
    extracted_1 = json.loads(f.read())
opfile = f"{os.getenv('PDIR')}/extracted/{args.model_id_2}/{args.label}.json"
with open(opfile, "r") as f:
    extracted_2 = json.loads(f.read())

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
ax_original.set_axis_off()
imgplot = ax_original.imshow(img, zorder=10)


years = [int(x) for x in csv["Years"]]
years = sorted(years)

# First model in the middle
ax_model_1 = fig.add_axes([0.35, 0.13, 0.31, 0.63])
ax_model_1.set_xlim(years[0] - 0.5, years[-1] + 0.5)
ax_model_1.set_xticks(range(years[0], years[-1] + 1))
ax_model_1.set_xticklabels(years)
ax_model_1.set_ylim(0.5, 13)
ax_model_1.set_yticks(range(1, 13))
ax_model_1.set_yticklabels(
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
ax_model_1.xaxis.set_ticks_position("top")
ax_model_1.xaxis.set_label_position("top")
ax_model_1.invert_yaxis()
ax_model_1.set_aspect("auto")

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


# Present extracted data as a %.2f string as far as possible
def format_value(value):
    if value is None or value == "null":
        return "null"
    try:
        return "%.2f" % float(value)
    except ValueError:
        return str(value)


for year in years:
    for month in monthNumbers.keys():
        try:
            exv = format_value(extracted_1["%s" % (year)][month])
            rrv = format_value(csv[month][year - min(years)])
            try:
                if exv == rrv:
                    ax_model_1.text(
                        year,
                        monthNumbers[month],
                        exv,
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="black",
                    )
                else:
                    ax_model_1.text(
                        year,
                        monthNumbers[month],
                        exv,
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="red",
                    )
                    ax_model_1.text(
                        year,
                        monthNumbers[month] + 0.5,
                        rrv,
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="blue",
                    )
            except Exception as e:
                print(rrv, exv)
                print(e)
        except KeyError as e:
            continue

# Second model on the right
ax_model_2 = fig.add_axes([0.67, 0.13, 0.31, 0.63])
ax_model_2.set_xlim(years[0] - 0.5, years[-1] + 0.5)
ax_model_2.set_xticks(range(years[0], years[-1] + 1))
ax_model_2.set_xticklabels(years)
ax_model_2.set_ylim(0.5, 13)
ax_model_2.set_yticks(range(1, 13))
ax_model_2.set_yticklabels(
    (
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    )
)
ax_model_2.xaxis.set_ticks_position("top")
ax_model_2.xaxis.set_label_position("top")
ax_model_2.invert_yaxis()
ax_model_2.set_aspect("auto")


for year in years:
    for month in monthNumbers.keys():
        try:
            exv = format_value(extracted_2["%s" % (year)][month])
            rrv = format_value(csv[month][year - min(years)])
            try:
                if exv == rrv:
                    ax_model_2.text(
                        year,
                        monthNumbers[month],
                        exv,
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="black",
                    )
                else:
                    ax_model_2.text(
                        year,
                        monthNumbers[month],
                        exv,
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="red",
                    )
                    ax_model_2.text(
                        year,
                        monthNumbers[month] + 0.5,
                        rrv,
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="blue",
                    )
            except Exception as e:
                print(rrv, exv)
                print(e)
        except KeyError as e:
            continue

# Render
fig.savefig(
    "compare_%s.webp" % (args.label),
)
