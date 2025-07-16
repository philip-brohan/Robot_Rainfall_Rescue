# Functions to make validation plots for rainfall_rescue


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
