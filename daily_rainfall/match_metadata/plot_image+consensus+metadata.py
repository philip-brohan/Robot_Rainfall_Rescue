#!/usr/bin/env python

# Plot a Daily Rainfall image along with consensus data from all models for comparison
# Look up the monthly averages in the RR dataset to get metadata about the station.
# Plot the monthly data comparison as well.

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
)
from daily_rainfall.match_metadata.compare_RR import (
    get_RR_monthly_db,
    search_RR_monthly_db,
)

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from cartopy import crs as ccrs
from cartopy.io import shapereader


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", action="append", default=[], help="Names of models to use"
)
parser.add_argument(
    "--image",
    help="Image identifier",
    type=str,
    required=True,
)
parser.add_argument(
    "--top_k",
    help="Number of top matching RR stations to display",
    type=int,
    required=False,
    default=3,
)
args = parser.parse_args()
if len(args.model) < 2:
    raise ValueError("Please provide at least two model IDs using --model")

# load the image
img = load_image(image_id_to_filename(args.image))

# Load the model extracted data
extracted = []
for model_id in args.model:
    fname = image_id_to_transcription_filename(args.image, group=model_id)
    try:
        data = load_json(fname)
        extracted.append(data)
    except Exception:
        print(f"Failed load of {fname}")
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

# Find matching RR stations
RRdb = get_RR_monthly_db()
monthly_averages = get_consensus_monthly_averages(extracted)
RR_results = search_RR_monthly_db(RRdb, monthly_averages, top_k=args.top_k)

# Look up those stations' metadata
# Load station metadata from file
import os
import json5 as json  # json5 can handle trailing commas

meta_file = os.path.join(os.getenv("PDIR"), "station_metadata.json")
with open(meta_file, mode="r") as file:
    station_meta = json.load(file)

# Add two stacked panels top-right: top = consensus + RR monthly averages,
# bottom = differences (RR - consensus)
region_x, region_y, region_w, region_h = 0.72, 0.60, 0.27, 0.35
# allocate top and bottom heights within the region
top_h = 0.20
bottom_h = 0.15
top_y = region_y + bottom_h
bottom_y = region_y

ax_monthly_top = fig.add_axes([region_x, top_y, region_w, top_h])
ax_monthly_bot = fig.add_axes([region_x, bottom_y, region_w, bottom_h])

months = list(range(1, 13))
consensus_averages = monthly_averages
# Top panel: consensus + RR
ax_monthly_top.plot(
    months,
    consensus_averages,
    marker="o",
    label="Consensus",
    color="blue",
    zorder=10,
)
# collect legend by setting labels on plotted lines
for res in RR_results[0]:
    rr_monthly_averages = res.entity.get("monthly_averages")
    station_name = res.entity.get("station_name")
    station_number = res.entity.get("station_number")
    year = res.entity.get("year")
    # distance may be available as `distance` or `score` depending on client
    distance = None
    if hasattr(res, "distance"):
        distance = getattr(res, "distance")
    elif hasattr(res, "score"):
        distance = getattr(res, "score")
    # Format label as three fixed-width columns: distance (3 dp), year, station name
    if distance is None:
        dist_str = "   -   "
    else:
        try:
            dist_str = f"{float(distance):7.3f}"
        except Exception:
            dist_str = str(distance)
    label = f"{dist_str}  {int(year):4d}  {station_name}"
    ax_monthly_top.plot(
        months,
        rr_monthly_averages,
        marker="o",
        label=label,
        linestyle="--",
        zorder=5,
    )

# Top panel formatting
ax_monthly_top.set_title("Monthly Averages Comparison")
ax_monthly_top.set_ylabel("Monthly Total (mm)")
ax_monthly_top.set_xticks([])  # hide x labels on top panel

# Create legend below the bottom panel using the top-panel handles/labels
handles, labels = ax_monthly_top.get_legend_handles_labels()
ax_monthly_bot.legend(
    handles=handles,
    labels=labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.35),
    bbox_transform=ax_monthly_bot.transAxes,
    ncol=1,
    fontsize="small",
    prop={"family": "monospace"},
    title="  dist    year  station",
)

# Bottom panel: differences RR - consensus
for res in RR_results[0]:
    rr_monthly_averages = res.entity.get("monthly_averages")
    # compute difference robustly
    diff = []
    for i in range(12):
        try:
            r = (
                float(rr_monthly_averages[i])
                if rr_monthly_averages and rr_monthly_averages[i] is not None
                else 0.0
            )
        except Exception:
            r = 0.0
        try:
            c = (
                float(consensus_averages[i])
                if consensus_averages and consensus_averages[i] is not None
                else 0.0
            )
        except Exception:
            c = 0.0
        diff.append(r - c)
    ax_monthly_bot.plot(months, diff, marker="s", linestyle="-", zorder=5)

# Bottom panel formatting
ax_monthly_bot.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
ax_monthly_bot.set_xticks(months)
ax_monthly_bot.set_xticklabels(
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
ax_monthly_bot.set_xlabel("Month")
ax_monthly_bot.set_ylabel("RR - Consensus (mm)")


# Add an outline map of the United Kingdom in the bottom-right using Cartopy

map_ax = fig.add_axes([0.6, 0.02, 0.5, 0.35], projection=ccrs.PlateCarree())
map_ax.set_extent([-11, 4, 49, 61], crs=ccrs.PlateCarree())
# Draw the UK outline from Natural Earth admin_0_countries
shp = shapereader.natural_earth(
    resolution="10m", category="cultural", name="admin_0_countries"
)
reader = shapereader.Reader(shp)
for rec in reader.records():
    admin = rec.attributes.get("ADMIN") or rec.attributes.get("NAME")
    iso = rec.attributes.get("ISO_A3")
    if admin == "United Kingdom" or iso == "GBR":
        geom = rec.geometry
        map_ax.add_geometries(
            [geom],
            ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="black",
            linewidth=0.9,
        )
        break
map_ax.set_xticks([])
map_ax.set_yticks([])
map_ax.set_aspect("equal")
# Plot the station location(s) from the top RR results
count = 0
for res in RR_results[0]:
    station_number = res.entity.get("station_number")
    if station_number in station_meta:
        lat = station_meta[station_number].get("lat")
        long = station_meta[station_number].get("long")
        if lat is not None and long is not None:
            map_ax.plot(
                long,
                lat,
                marker="o",
                color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][
                    count % 5
                ],  # Match colours above - from the tab10 colour cycle, matplotlib default
                markersize=8,
                markeredgecolor="black",
                transform=ccrs.PlateCarree(),
                zorder=20,
            )
        count += 1

# Add top-centre title showing the image identifier, inserting newlines after '/'
title = str(args.image).replace("/", "/\n")
try:
    fig.suptitle(title, x=0.5, y=0.95, ha="center", va="top", fontsize="x-large")
except Exception:
    fig.text(0.5, 0.95, title, ha="center", va="top", fontsize="x-large")

# Render
fig.savefig("consensus+metadata.webp")
