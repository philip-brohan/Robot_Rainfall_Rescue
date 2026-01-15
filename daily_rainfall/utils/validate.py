# Functions to make validation plots for daily rainfall

from sympy import group
from rainfall_rescue.utils.pairs import load_pair, csv_to_json
import re
import json
import os
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
import numpy as np
from collections import Counter

# Map month names to numbers
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


# Plot the image into a given axes
def plot_image(ax, img):
    ax.set_axis_off()
    imgplot = ax.imshow(img, zorder=10)


def plot_daily_table(ax_digitised, dd1, group=None, comparison=None):
    #    ax_digitised = fig.add_axes([0.52, 0.13, 0.47, 0.63])
    ax_digitised.set_xlim(0.5, 12.5)
    ax_digitised.set_xticks(range(1, 13))
    ax_digitised.set_xticklabels(
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
    ax_digitised.set_ylim(0, 32)
    ax_digitised.set_yticks(range(1, 32))
    ax_digitised.set_yticklabels(range(1, 32))
    ax_digitised.invert_yaxis()
    ax_digitised.set_aspect("auto")

    if group == "Gemini3":
        for month in dd1["Month"]:
            for day in month["rainfall"]:
                ax_digitised.text(
                    monthNumbers[month["Month"][:3]],
                    day["Day"],
                    day["rainfall"],
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="black",
                )
    else:
        for day in range(1, 32):
            day_key = f"Day {day}"
            if day_key in dd1:
                for month_idx in range(12):
                    val = dd1[day_key][month_idx]
                    if val is not None:
                        colour = "black"
                        if val == "null":
                            colour = "grey"
                        if (
                            comparison is not None
                            and comparison[day_key][month_idx] is not None
                        ):
                            comp_val = comparison[day_key][month_idx]
                            if val == comp_val:
                                if val == "null":
                                    colour = "lightblue"
                                else:
                                    colour = "blue"
                            else:
                                if val == "null":
                                    colour = "red"
                                else:
                                    colour = "red"
                        ax_digitised.text(
                            month_idx + 1,
                            day,
                            val,
                            ha="center",
                            va="center",
                            fontsize=12,
                            color=colour,
                        )


# Totals along the bottom
def plot_totals(ax_totals, dd1, group=None, comparison=None):
    ax_totals.set_xlim(0.5, 12.5)
    ax_totals.set_ylim(0, 1)
    ax_totals.set_xticks([])
    ax_totals.set_yticks([])

    if group != "Gemini3":
        for month_idx in range(12):
            val = dd1["Totals"][month_idx]
            if val is not None:
                colour = "black"
                if val == "null":
                    colour = "grey"
                if (
                    comparison is not None
                    and comparison["Totals"][month_idx] is not None
                ):
                    comp_val = comparison["Totals"][month_idx]
                    if val == comp_val:
                        if val == "null":
                            colour = "light blue"
                        else:
                            colour = "blue"
                    else:
                        if val == "null":
                            colour = "light red"
                        else:
                            colour = "red"
                ax_totals.text(
                    month_idx + 1,
                    0.5,
                    val,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=colour,
                )
    else:
        for month in dd1["Month"]:
            midx = monthNumbers[month["Month"][:3]] - 1
            val = month["total"]
            ax_totals.text(
                midx + 1,
                0.5,
                val,
                ha="center",
                va="center",
                fontsize=12,
                color="black",
            )
