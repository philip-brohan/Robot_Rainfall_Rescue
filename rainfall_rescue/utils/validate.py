# Functions to make validation plots for rainfall_rescue

from rainfall_rescue.utils.pairs import load_pair, csv_to_json
import re
import json
import os
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
import numpy as np
from collections import Counter


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


# Present extracted data as a %.2f string as far as possible
def format_value(data, key, year_idx):
    if key == "Name" or key == "Number":
        try:
            return data[key]
        except KeyError:
            return "N/A"
    if key == "Years":
        try:
            return str(data[key][year_idx])
        except (IndexError, KeyError):
            return "N/A"
    try:
        value = data[key][year_idx]
    except (IndexError, KeyError):
        return "N/A"

    return format_as_2f(value)


def format_as_2f(value):
    """Format a value as a string with two decimal places."""
    if value is None or value == "null":
        return "null"
    try:
        return "%.2f" % float(value)
    except ValueError:
        return str(value)  # Return as is if it cannot be converted to float


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


def models_agree(extracted, value, idx=None, agreement_count=2):
    """Check if the models agree on a value."""
    values = []
    for model_id in extracted.keys():
        if idx is not None:
            if value == "Years":
                val = extracted[model_id][value][idx]
            else:
                val = format_as_2f(extracted[model_id][value][idx])
        else:
            val = extracted[model_id][value]
        values.append(val)
    counts = Counter(values)
    top_two = counts.most_common(2)
    if top_two[0][0] == "N/A":  # Special case - 'agreed' on "can't do it"
        return (False, "N/A")  # Not counted as agreement
    if len(top_two) < 2:
        return (True, top_two[0][0])  # Only one unique value
    if top_two[0][1] == top_two[1][1]:  # No one most common value
        return (False, top_two[0][0])
    if top_two[0][1] >= agreement_count:  # Most common value is popular enough
        return (True, top_two[0][0])
    return (False, top_two[0][0])


def plot_metadata_agreement(ax, extracted, jcsv, agreement_count=2):

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    ymp = 0.8
    for metad in ("Number", "Name"):
        match, exv = models_agree(extracted, metad, agreement_count=agreement_count)
        rrv = jcsv[metad]
        if match:  # Models agree
            if exv == rrv:  # on the right answer
                colour = (0, 0, 1)  # Blue
            else:  # on the wrong answer
                colour = (1, 0, 0)  # Red
        else:  # Models disagree
            colour = (0.5, 0.5, 0.5)  # Grey
        ax.text(
            0.05,
            ymp,
            "%s: %s" % (metad, exv),
            fontsize=14,
            color=colour,
        )
        ymp -= 0.3


# Plot fractional success at metadata into a given axes
def plot_metadata_fraction_agreement(ax, merged, cmp=None):

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    ymp = 0.8
    for metad in ("Number", "Name"):
        blue = merged[metad].count("blue") / len(merged[metad])
        plot_two_colored_text(
            ax,
            0.05,
            ymp,
            "%s: " % metad,
            "  %d" % int((blue) * 100),
            size=12,
            colour1="black",
            colour2="blue",
        )
        red = merged[metad].count("red") / len(merged[metad])
        if int(red * 100) > 0:
            plot_two_colored_text(
                ax,
                0.05,
                ymp - 0.15,
                "%s: " % metad,
                "  %d" % int((red) * 100),
                size=12,
                colour1="white",  # Invisible
                colour2="red",
            )

        ymp -= 0.3


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
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.xaxis.set_ticks_position("top")
    labels = ax.set_xticklabels(extracted["Years"])
    # Note - this has to be the last change made to the xtics, or the colours will be reset
    for year_idx, label in enumerate(labels):
        if extracted["Years"][year_idx] != jcsv["Years"][year_idx]:
            label.set_color("red")
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

    for year_idx in range(10):
        for month in monthNumbers.keys():
            try:
                exv = format_value(extracted, month, year_idx)
                rrv = format_value(jcsv, month, year_idx)
                try:
                    if exv == rrv:
                        ax.text(
                            year_idx + 1,
                            monthNumbers[month],
                            exv,
                            ha="center",
                            va="center",
                            fontsize=12,
                            color="black",
                        )
                    else:
                        ax.text(
                            year_idx + 1,
                            monthNumbers[month],
                            exv,
                            ha="center",
                            va="center",
                            fontsize=12,
                            color="red",
                        )
                        ax.text(
                            year_idx + 1,
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


def plot_monthly_table_agreement(ax, extracted, jcsv, agreement_count=2, yticks=True):
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    xtl = ["N/A"] * 10
    for year_idx in range(10):
        match, xtl[year_idx] = models_agree(
            extracted, "Years", idx=year_idx, agreement_count=agreement_count
        )
    labels = ax.set_xticklabels(xtl)
    for year_idx, label in enumerate(labels):
        match, exv = models_agree(
            extracted, "Years", idx=year_idx, agreement_count=agreement_count
        )
        if match:  # Models agree
            if exv == jcsv["Years"][year_idx]:
                label.set_color("blue")
            else:  # on the wrong answer
                label.set_color("red")
        else:  # Models disagree
            label.set_color("grey")
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

    for year_idx in range(10):
        for month in monthNumbers.keys():
            try:
                match, exv = models_agree(
                    extracted, month, idx=year_idx, agreement_count=agreement_count
                )
                if match:  # Models agree
                    if exv == jcsv[month][year_idx]:  # on the right answer
                        colour = (0, 0, 1)  # Blue
                    else:  # on the wrong answer
                        colour = (1, 0, 0)  # Red
                else:  # Models disagree
                    colour = (0.5, 0.5, 0.5)
                ax.text(
                    year_idx + 1,
                    monthNumbers[month],
                    exv,
                    ha="center",
                    va="center",
                    fontsize=14,
                    color=colour,
                )
            except KeyError as e:
                continue


def plot_monthly_table_fraction_agreement(ax, merged, cmp=None, yticks=True):
    years = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xticks(range(years[0], years[-1] + 1))
    xtfraction = [
        merged["Years"][year_idx].count("blue") / len(merged["Years"][year_idx])
        for year_idx in range(10)
    ]
    xtl = [f"{int(fraction * 100)}" for fraction in xtfraction]
    ax.set_xticklabels(xtl)
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
            blue = merged[month][year - 1].count("blue") / len(merged[month][year - 1])
            ax.text(
                year,
                monthNumbers[month],
                f"{int(blue * 100)}",
                ha="center",
                va="center",
                fontsize=12,
                color="blue",
            )
            red = merged[month][year - 1].count("red") / len(merged[month][year - 1])
            if (int(red * 100)) > 0:
                ax.text(
                    year,
                    monthNumbers[month] + 0.5,
                    f"{int(red * 100)}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="red",
                )


def plot_monthly_table_fraction(ax, merged, cmp=None, yticks=True):
    years = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xticks(range(years[0], years[-1] + 1))
    xtfraction = [
        sum(merged["Years"][year_idx]) / len(merged["Years"][year_idx])
        for year_idx in range(10)
    ]
    xtl = [f"{int(fraction * 100)}" for fraction in xtfraction]
    ax.set_xticklabels(xtl)
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
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    for year_idx in range(10):
        exv = format_value(extracted, "Totals", year_idx)
        rrv = format_value(jcsv, "Totals", year_idx)
        if exv == rrv:
            ax.text(
                year_idx + 1,
                0.7,
                exv,
                ha="center",
                va="center",
                fontsize=12,
                color="black",
            )
        else:
            ax.text(
                year_idx + 1,
                0.7,
                exv,
                ha="center",
                va="center",
                fontsize=12,
                color="red",
            )
            ax.text(
                year_idx + 1,
                0.3,
                rrv,
                ha="center",
                va="center",
                fontsize=12,
                color="blue",
            )


# Mark where multi models agreed - for totals
def plot_totals_agreement(
    ax,
    extracted,
    jcsv,
    agreement_count=2,
):
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    for year_idx in range(0, 10):
        match, exv = models_agree(
            extracted, "Totals", idx=year_idx, agreement_count=agreement_count
        )
        rrv = format_value(jcsv, "Totals", year_idx)

        if match:  # Models agree
            if exv == rrv:  # on the right answer
                colour = (0, 0, 1)  # Blue
            else:  # on the wrong answer
                colour = (1, 0, 0)  # Red
        else:  # Models disagree
            colour = (0.5, 0.5, 0.5)  # Grey
        ax.text(
            year_idx + 1,
            0.5,
            exv,
            ha="center",
            va="center",
            fontsize=14,
            color=colour,
        )


def plot_totals_fraction_agreement(ax, merged, cmp=None):
    years = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    for year in years:
        blue = merged["Totals"][year - 1].count("blue") / len(
            merged["Totals"][year - 1]
        )
        ax.text(
            year,
            0.7,
            f"{int(blue * 100)}",
            ha="center",
            va="center",
            fontsize=12,
            color="blue",
        )
        red = merged["Totals"][year - 1].count("red") / len(merged["Totals"][year - 1])
        if (int(red * 100)) > 0:
            ax.text(
                year,
                0.3,
                f"{int(red * 100)}",
                ha="center",
                va="center",
                fontsize=12,
                color="red",
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


def make_null_json():
    """Create a null JSON object for cases where no data is available."""
    all_na = {"Name": "N/A", "Number": "N/A", "Years": ["N/A"] * 10}
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
        all_na[month] = ["N/A"] * 10
    all_na["Totals"] = ["N/A"] * 10
    return all_na


# Models don't always make good JSON - fix the egregious problems so it parses
def quote_list_items(match):
    # Get the content inside the brackets
    items = [v.strip() for v in match.group(1).split(",")]
    quoted = ['"%s"' % v for v in items]
    return "[" + ", ".join(quoted) + "]"


def jsonfix(input):
    """Fix JSON that has numbers like .12 instead of 0.12"""
    fixed = re.sub(r"(?<!\d)\.(\d+)", r"0.\1", input)  # Fix numbers like .12 -> 0.12
    fixed = re.sub(r"(\d+):", r'"\1":', fixed)  # Fix keys like 2023: -> "2023":
    fixed = re.sub(r"\[([^\[\]]+)\]", quote_list_items, fixed)  # Quote list items
    fixed = fixed.replace('""', '"')
    fixed = "".join(
        c for c in fixed if c.isprintable()
    )  # Get rid of line breaks and other non-printable characters
    # Deal with bad terminations
    if not fixed.endswith("]}"):
        fixed = (
            fixed[: fixed.rfind("]") + 1] + "}"
        )  # Might cut off too much, but if so we're screwed anyway.
    # Get rid of any junk after the totals
    last_match = None
    for m in re.finditer(r'"Totals"\s*:\s*\[.*?\]', fixed, flags=re.DOTALL):
        last_match = m
    fixed = fixed[: last_match.end()] + "}" if last_match else fixed
    return fixed


# map JSON jeys like 'TOTALS' to 'Totals'
def cap_first_key(k: str) -> str:
    return k[:1].upper() + k[1:] if isinstance(k, str) else k


# Load the extracted data from a model, for a label
def load_extracted(model_id, label):
    """Load the extracted data from a model for a given label."""
    null_j = make_null_json()
    opfile = f"{os.getenv('PDIR')}/extracted/{model_id}/{label}.json"
    if not os.path.exists(opfile):
        print(f"No extraction for {model_id} {label}")
        return null_j
    with open(opfile, "r") as f:
        raw_j = f.read()
        fixed_j = jsonfix(raw_j)
        try:
            extracted = json.loads(fixed_j)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for {model_id} {label}: {e}")
            return null_j
    for key in null_j:
        if key not in extracted:
            extracted[key] = null_j[key]
        elif isinstance(extracted[key], list) and len(extracted[key]) < 10:
            # Ensure lists have 10 items
            extracted[key] += ["N/A"] * (10 - len(extracted[key]))
    # Fix boring common error
    if extracted["Name"].lower().startswith("rainfall at"):
        extracted["Name"] = extracted["Name"][12:]
    extracted = {cap_first_key(k): v for k, v in extracted.items()}
    return extracted


# find where the model is accurate for each value in one case
def validate_case(model_id, label):

    # load the image/data pair
    img, csv = load_pair(label)
    jcsv = json.loads(csv_to_json(csv))

    # Load the model extracted data
    extracted = load_extracted(model_id, label)

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
