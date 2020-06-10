#!/usr/bin/env python

# Make a bare-bones imitation of a ten-year rainfall sheet
#  use random data.

import os
import random
import pickle
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Specify where to put the output
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--docn", help="Document name", type=str, required=True)
parser.add_argument(
    "--opdir", help="output directory name", default=None, type=str, required=False
)
args = parser.parse_args()
if args.opdir is None:
    args.opdir = ("%s/ML_ten_year_rainfall/fakes/plain") % os.getenv("SCRATCH")
if not os.path.isdir("%s/images" % args.opdir):
    os.makedirs("%s/images" % args.opdir)
if not os.path.isdir("%s/numbers" % args.opdir):
    os.makedirs("%s/numbers" % args.opdir)

# Set the parameters of the figure
# Alter these to mimic the variability in the real image
imp = {
    "scale": 1.0,
    "xscale": 1.0,
    "yscale": 1.0,
    "xshift": 0.0,  # pixels, +ve right
    "yshift": 0.0,  # pixels, +ve up
    "rotate": 0.0,  # degrees clockwise
    "linewidth": 1.0,
    "bgcolour": (1.0, 1.0, 1.0),
    "fgcolour": (0.0, 0.0, 0.0),
    "yearHeight": 0.066,  # Fractional height of year row
    "totalsHeight": 0.105,  # Fractional height of totals row
    "monthsWidth": 0.137,  # Fractional width of months row
    "meansWidth": 0.107,  # Fractional width of means row
    "fontSize": 10,
    "year": 1941,
}

# Figure is 768x1024
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
ax_full = fig.add_axes([0, 0, 1, 1])
ax_full.set_xlim([0, 1])
ax_full.set_ylim([0, 1])
ax_full.set_axis_off()

# Paint the background white - why is this needed?
ax_full.add_patch(
    matplotlib.patches.Rectangle((0, 0), 1, 1, fill=True, facecolor="white")
)

# Box with the data in
topLeft = (0.07 + imp["xshift"] / 768, 0.725 + imp["yshift"] / 1024)
topRight = (
    0.93 + imp["xshift"] / 768 + (imp["xscale"] - 1) * 0.86,
    0.725 + imp["yshift"] / 1024,
)
bottomLeft = (0.07 + imp["xshift"] / 768, 0.325 + imp["yshift"] / 1024)
bottomRight = (
    0.93 + imp["xshift"] / 768 + (imp["xscale"] - 1) * 0.86,
    0.325 + imp["yshift"] / 1024 - (imp["yscale"] - 1) * 0.4,
)
ax_full.add_line(
    Line2D(
        xdata=(topLeft[0], topRight[0], bottomRight[0], bottomLeft[0], topLeft[0]),
        ydata=(topLeft[1], topRight[1], bottomRight[1], bottomLeft[1], topLeft[1]),
        linestyle="solid",
        linewidth=imp["linewidth"],
        color=imp["fgcolour"],
        zorder=1,
    )
)


def topAt(x):  # x is fraction along top line
    return (
        topRight[0] * x + topLeft[0] * (1 - x),
        topRight[1] * x + topLeft[1] * (1 - x),
    )


def bottomAt(x):
    return (
        bottomRight[0] * x + bottomLeft[0] * (1 - x),
        bottomRight[1] * x + bottomLeft[1] * (1 - x),
    )


def leftAt(y):  # y is fraction of way from bottom to top
    return (
        topLeft[0] * y + bottomLeft[0] * (1 - y),
        topLeft[1] * y + bottomLeft[1] * (1 - y),
    )


def rightAt(y):
    return (
        topRight[0] * y + bottomRight[0] * (1 - y),
        topRight[1] * y + bottomRight[1] * (1 - y),
    )


# Draw the grid
lft = leftAt(1.0 - imp["yearHeight"])
rgt = rightAt(1.0 - imp["yearHeight"])
ax_full.add_line(
    Line2D(
        xdata=(lft[0], rgt[0]),
        ydata=(lft[1], rgt[1]),
        linestyle="solid",
        linewidth=imp["linewidth"],
        color=imp["fgcolour"],
        zorder=1,
    )
)
lft = leftAt(imp["totalsHeight"])
rgt = rightAt(imp["totalsHeight"])
ax_full.add_line(
    Line2D(
        xdata=(lft[0], rgt[0]),
        ydata=(lft[1], rgt[1]),
        linestyle="solid",
        linewidth=imp["linewidth"],
        color=imp["fgcolour"],
        zorder=1,
    )
)
tp = topAt(imp["monthsWidth"])
bm = bottomAt(imp["monthsWidth"])
ax_full.add_line(
    Line2D(
        xdata=(tp[0], bm[0]),
        ydata=(tp[1], bm[1]),
        linestyle="solid",
        linewidth=imp["linewidth"],
        color=imp["fgcolour"],
        zorder=1,
    )
)
tp = topAt(1.0 - imp["meansWidth"])
bm = bottomAt(1.0 - imp["meansWidth"])
ax_full.add_line(
    Line2D(
        xdata=(tp[0], bm[0]),
        ydata=(tp[1], bm[1]),
        linestyle="solid",
        linewidth=imp["linewidth"],
        color=imp["fgcolour"],
        zorder=1,
    )
)
for yrl in range(1, 10):
    x = imp["monthsWidth"] + yrl * (1.0 - imp["meansWidth"] - imp["monthsWidth"]) / 10
    tp = topAt(x)
    bm = bottomAt(x)
    ax_full.add_line(
        Line2D(
            xdata=(tp[0], bm[0]),
            ydata=(tp[1], bm[1]),
            linestyle="solid",
            linewidth=imp["linewidth"],
            color=imp["fgcolour"],
            zorder=1,
        )
    )

# Add the fixed text
tp = topAt(imp["monthsWidth"] / 2)
lft = leftAt(1.0 - imp["yearHeight"] / 2)
ax_full.text(
    tp[0],
    lft[1],
    "Year",
    fontsize=imp["fontSize"],
    horizontalalignment="center",
    verticalalignment="center",
)
tp = topAt(1.0 - imp["meansWidth"] / 2)
lft = leftAt(1.0 - imp["yearHeight"] / 2)
ax_full.text(
    tp[0],
    lft[1],
    "Means",
    fontsize=imp["fontSize"],
    horizontalalignment="center",
    verticalalignment="center",
)
tp = topAt(imp["monthsWidth"] / 2)
lft = leftAt(imp["totalsHeight"] / 2)
ax_full.text(
    tp[0],
    lft[1],
    "Totals",
    fontsize=imp["fontSize"],
    horizontalalignment="center",
    verticalalignment="center",
)
months = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
tp = topAt(imp["monthsWidth"] / 10)
for mdx in range(len(months)):
    lft = leftAt(
        1.0
        - imp["yearHeight"]
        - (mdx + 1)
        * (1.0 - imp["yearHeight"] - imp["totalsHeight"])
        / (len(months) + 1)
    )
    ax_full.text(
        tp[0],
        lft[1],
        months[mdx],
        fontsize=imp["fontSize"] - 1,
        horizontalalignment="left",
        verticalalignment="center",
    )
lft = leftAt(1.0 - imp["yearHeight"] / 2)
for ydx in range(10):
    x = (
        imp["monthsWidth"]
        + (ydx + 0.5) * (1.0 - imp["meansWidth"] - imp["monthsWidth"]) / 10
    )
    tp = topAt(x)
    ax_full.text(
        tp[0],
        lft[1],
        "%04d" % (imp["year"] + ydx),
        fontsize=imp["fontSize"],
        horizontalalignment="center",
        verticalalignment="center",
    )

# Generate random numbers to fill out the data table
#  Each month's data is represented by 3 integers (0-9) - x, y, and z
#  where the accumulated rainfall for that month is x.yz inches.
rdata = []
for yri in range(10):
    ydata = []
    for mni in range(12):
        mdata = []
        for pni in range(3):
            mdata.append(random.randint(0, 9))
        ydata.append(mdata)
    rdata.append(ydata)

# Store the data
with open("%s/numbers/%s.pkl" % (args.opdir, args.docn), "wb") as pf:
    pickle.dump(rdata, pf)

# Fill out the table with the random numbers
for yri in range(10):
    x = (
        imp["monthsWidth"]
        + (yri + 0.5) * (1.0 - imp["meansWidth"] - imp["monthsWidth"]) / 10
    )
    tp = topAt(x)
    for mni in range(12):
        lft = leftAt(
            1.0
            - imp["yearHeight"]
            - (mni + 1)
            * (1.0 - imp["yearHeight"] - imp["totalsHeight"])
            / (len(months) + 1)
        )
        inr = rdata[yri][mni][0] + rdata[yri][mni][1] / 10 + rdata[yri][mni][2] / 100
        ax_full.text(
            tp[0],
            lft[1],
            "%4.2f" % inr,
            fontsize=imp["fontSize"],
            horizontalalignment="center",
            verticalalignment="center",
        )

# Add the monthly means
tp = topAt(1.0 - imp["meansWidth"] / 2)
for mni in range(12):
    lft = leftAt(
        1.0
        - imp["yearHeight"]
        - (mni + 1)
        * (1.0 - imp["yearHeight"] - imp["totalsHeight"])
        / (len(months) + 1)
    )
    inr = 0.0
    for yri in range(10):
        inr += rdata[yri][mni][0] + rdata[yri][mni][1] / 10 + rdata[yri][mni][2] / 100
    inr /= 10
    ax_full.text(
        tp[0],
        lft[1],
        "%4.2f" % inr,
        fontsize=imp["fontSize"],
        horizontalalignment="center",
        verticalalignment="center",
    )

# Add the annual totals
lft = leftAt(imp["totalsHeight"] / 2)
for yri in range(10):
    x = (
        imp["monthsWidth"]
        + (yri + 0.5) * (1.0 - imp["meansWidth"] - imp["monthsWidth"]) / 10
    )
    tp = topAt(x)
    inr = 0.0
    for mni in range(12):
        inr += rdata[yri][mni][0] + rdata[yri][mni][1] / 10 + rdata[yri][mni][2] / 100
    ax_full.text(
        tp[0],
        lft[1],
        "%5.2f" % inr,
        fontsize=imp["fontSize"],
        horizontalalignment="center",
        verticalalignment="center",
    )


fig.savefig("%s/images/%s.png" % (args.opdir, args.docn))
