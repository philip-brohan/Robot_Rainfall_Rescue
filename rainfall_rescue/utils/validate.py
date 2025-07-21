# Functions to make validation plots for rainfall_rescue

from rainfall_rescue.utils.pairs import load_pair, csv_to_json
import re
import json
import os
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
import numpy as np


def plot_two_colored_text(
    ax, x, y, text1, text2, size=12, colour1="blue", colour2="red"
):

    # Calculate scaling factors from points to axes coordinates
    fig = ax.figure
    axes_bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    axes_width_inch = axes_bbox.width
    axes_height_inch = axes_bbox.height
    scale_x = 1 / axes_width_inch * (1 / 72)
    scale_y = 1 / axes_height_inch * (1 / 72)

    path = TextPath((0, 0), text1, size=size)
    trans = Affine2D().scale(scale_x, scale_y).translate(x, y) + ax.transAxes
    patch = PathPatch(path, color=colour1, linewidth=0, transform=trans)
    ax.add_patch(patch)
    verts = patch.get_path().vertices
    trans = patch.get_transform()
    verts_data = trans.transform(verts)
    verts_axes = ax.transAxes.inverted().transform(verts_data)
    right_limit_text1 = np.max(verts_axes[:, 0])
    path = TextPath((0, 0), text2, size=size)
    trans = (
        Affine2D().scale(scale_x, scale_y).translate(right_limit_text1, y)
        + ax.transAxes
    )
    patch = PathPatch(path, color=colour2, linewidth=0, transform=trans)
    ax.add_patch(patch)


# Get an integer set of years
def get_years(jcsv):
    years = [int(x) for x in jcsv["Years"]]
    years = sorted(years)
    return years


# Present extracted data as a %.2f string as far as possible
def format_value(data, month, year):
    try:
        value = data[month][year - min(get_years(data))]
    except (IndexError, KeyError):
        return "N/A"
    if value is None or value == "null":
        return "null"
    try:
        return "%.2f" % float(value)
    except ValueError:
        return str(value)


# Plot the image into a given axes
def plot_image(ax, img):
    ax.set_axis_off()
    imgplot = ax.imshow(img, zorder=10)


# Plot target and retrieved image metadata into a given axes
def plot_metadata(ax, extracted, jcsv):

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    ymp = 0.8
    for metad in ("Number", "Name"):
        exv = extracted[metad]
        rrv = jcsv[metad]
        if exv == rrv:
            ax.text(
                0.05,
                ymp,
                "%s: %s" % (metad, exv),
                fontsize=12,
                color="black",
            )
        else:
            ax.text(
                0.05,
                ymp,
                "%s: %s" % (metad, exv),
                fontsize=12,
                color="red",
            )
            ax.text(
                0.05,
                ymp - 0.1,
                "%s: %s" % (metad, rrv),
                fontsize=12,
                color="blue",
            )
        ymp -= 0.3


# Plot fractiobnal success at metadata into a given axes
def plot_metadata_fraction(ax, merged, cmp=None):

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    ymp = 0.8
    for metad in ("Number", "Name"):
        fraction = sum(merged[metad]) / len(merged[metad])
        ax.text(
            0.05,
            ymp,
            "%s: %d" % (metad, int(fraction * 100)),
            fontsize=12,
            color="black",
        )
        if cmp is not None:
            cmp_fraction = sum(cmp[metad]) / len(cmp[metad])
            color = "blue" if cmp_fraction < fraction else "red"
            plot_two_colored_text(
                ax,
                0.05,
                ymp - 0.15,
                "%s: " % metad,
                "  %d" % abs(int((fraction - cmp_fraction) * 100)),
                size=12,
                colour1="white",  # Invisible
                colour2=color,
            )

        ymp -= 0.3


# Plot the digitised numbers into a given axes
def plot_monthly_table(ax, extracted, jcsv, yticks=True):
    years = get_years(jcsv)
    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.set_xticks(range(years[0], years[-1] + 1))
    ax.set_xticklabels(years)
    ax.set_ylim(0.5, 13)
    if yticks:
        ax.set_yticks(range(1, 13))
        ax.set_yticklabels(
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
    else:
        ax.set_yticks([])
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.invert_yaxis()
    ax.set_aspect("auto")

    monthNumbers = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }

    for year in years:
        for month in monthNumbers.keys():
            try:
                exv = format_value(extracted, month, year)
                rrv = format_value(jcsv, month, year)
                try:
                    if exv == rrv:
                        ax.text(
                            year,
                            monthNumbers[month],
                            exv,
                            ha="center",
                            va="center",
                            fontsize=12,
                            color="black",
                        )
                    else:
                        ax.text(
                            year,
                            monthNumbers[month],
                            exv,
                            ha="center",
                            va="center",
                            fontsize=12,
                            color="red",
                        )
                        ax.text(
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


def plot_monthly_table_fraction(ax, merged, cmp=None, yticks=True):
    years = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.set_xticks(range(years[0], years[-1] + 1))
    ax.set_xticklabels(years)
    ax.set_ylim(0.5, 13)
    if yticks:
        ax.set_yticks(range(1, 13))
        ax.set_yticklabels(
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
    else:
        ax.set_yticks([])
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.invert_yaxis()
    ax.set_aspect("auto")

    monthNumbers = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }

    for year in years:
        for month in monthNumbers.keys():
            fraction = sum(merged[month][year - 1]) / len(merged[month][year - 1])
            ax.text(
                year,
                monthNumbers[month],
                f"{int(fraction * 100)}",
                ha="center",
                va="center",
                fontsize=12,
                color="black",
            )
            if cmp is not None:
                cmp_fraction = sum(cmp[month][year - 1]) / len(cmp[month][year - 1])
                color = "blue" if cmp_fraction < fraction else "red"
                ax.text(
                    year,
                    monthNumbers[month] + 0.5,
                    "%d" % abs(int((fraction - cmp_fraction) * 100)),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=color,
                )


# plot the extracted totals into a given axes
def plot_totals(ax, extracted, jcsv):
    years = get_years(jcsv)
    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    for year in years:
        exv = format_value(extracted, "Totals", year)
        rrv = format_value(jcsv, "Totals", year)
        if exv == rrv:
            ax.text(
                year,
                0.7,
                exv,
                ha="center",
                va="center",
                fontsize=12,
                color="black",
            )
        else:
            ax.text(
                year,
                0.7,
                exv,
                ha="center",
                va="center",
                fontsize=12,
                color="red",
            )
            ax.text(
                year,
                0.3,
                rrv,
                ha="center",
                va="center",
                fontsize=12,
                color="blue",
            )


def plot_totals_fraction(ax, merged, cmp=None):
    years = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    for year in years:
        fraction = sum(merged["Totals"][year - 1]) / len(merged["Totals"][year - 1])
        ax.text(
            year,
            0.7,
            f"{int(fraction * 100)}",
            ha="center",
            va="center",
            fontsize=12,
            color="black",
        )
        if cmp is not None:
            cmp_fraction = sum(cmp["Totals"][year - 1]) / len(cmp["Totals"][year - 1])
            color = "blue" if cmp_fraction < fraction else "red"
            ax.text(
                year,
                0.3,
                "%d" % abs(int((fraction - cmp_fraction) * 100)),
                ha="center",
                va="center",
                fontsize=12,
                color=color,
            )


# find where the model is accurate for each value in one case
def validate_case(model_id, label):

    # load the image/data pair
    img, csv = load_pair(label)
    jcsv = json.loads(csv_to_json(csv))

    # Load the model extracted data
    opfile = f"{os.getenv('PDIR')}/extracted/{model_id}/{label}.json"
    with open(opfile, "r") as f:
        raw_j = f.read()
        fixed_j = re.sub(
            r"(?<!\d)\.(\d+)", r"0.\1", raw_j
        )  # Fix numbers like .12 -> 0.12
        fixed_j = re.sub(r"(\d+):", r'"\1":', fixed_j)  # Fix keys like 2023: -> "2023":
        try:
            extracted = json.loads(fixed_j)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for case {label}: {e}")
            return None

    # Check if the extracted data matches the CSV data
    correct = {}
    try:
        correct["Name"] = jcsv["Name"] == extracted["Name"]
    except KeyError:
        correct["Name"] = False
    try:
        correct["Number"] = jcsv["Number"] == extracted["Number"]
    except KeyError:
        correct["Number"] = False
    correct["Years"] = [False] * 10
    correct["Totals"] = [False] * 10
    for month in (
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
    ):
        correct[month] = [False] * 10
    for yr in range(10):
        try:
            correct["Years"][yr] = jcsv["Years"][yr] == extracted["Years"][yr]
        except (KeyError, IndexError):
            correct["Years"][yr] = False
        try:
            correct["Totals"][yr] = jcsv["Totals"][yr] == extracted["Totals"][yr]
        except (KeyError, IndexError):
            correct["Totals"][yr] = False
        for month in (
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
        ):
            try:
                correct[month][yr] = jcsv[month][yr] == extracted[month][yr]
            except (KeyError, IndexError):
                correct[month][yr] = False

    return correct


# Merge validated cases into a single dictionary
def merge_validated_cases(merged, case):
    if merged is None:
        merged = case
        for key in merged:
            if isinstance(merged[key], list):
                for i in range(len(merged[key])):
                    merged[key][i] = [case[key][i]]
            else:
                merged[key] = [case[key]]
    else:
        for key in case:
            if isinstance(case[key], list):
                for i in range(len(case[key])):
                    merged[key][i].append(case[key][i])
            else:
                merged[key].append(case[key])
    return merged
