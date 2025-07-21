#!/usr/bin/env python

# Plot stats from multi-case validation

from rainfall_rescue.utils.validate import (
    plot_metadata_fraction,
    plot_monthly_table_fraction,
    plot_totals_fraction,
    validate_case,
    merge_validated_cases,
)
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    help="Model ID",
    type=str,
    required=False,
    default="google/gemma-3-4b-it",
)
args = parser.parse_args()

# Find all the cases with extractions available from this model
opdir = f"{os.getenv('PDIR')}/extracted/{args.model_id}"
cases = os.listdir(opdir)
if not cases:
    raise ValueError(f"No cases found for model {args.model_id}")
cases = [c[:-5] for c in cases if c.endswith(".json")]


# Validate all cases
merged = None
for label in cases:
    #    print(f"Validating case {label}")
    case = validate_case(args.model_id, label)
    if case is not None:
        merged = merge_validated_cases(merged, case)


# Create the figure
fig = Figure(
    figsize=(7, 10),  # Width, Height (inches)
    dpi=100,
    facecolor=(0.95, 0.95, 0.95, 1),
    edgecolor=None,
    linewidth=0.0,
    frameon=True,
    subplotpars=None,
    tight_layout=None,
)
canvas = FigureCanvas(fig)


# Metadata top right
ax_metadata = fig.add_axes([0.07, 0.8, 0.91, 0.15])
plot_metadata_fraction(ax_metadata, merged)


# # Digitised numbers on the right
ax_digitised = fig.add_axes([0.07, 0.13, 0.91, 0.63])
plot_monthly_table_fraction(ax_digitised, merged)

# # Totals along the bottom
ax_totals = fig.add_axes([0.07, 0.05, 0.91, 0.07])
plot_totals_fraction(ax_totals, merged)

# Render
fig.savefig(
    "%s.webp" % (args.model_id.replace("/", "_")),
)
