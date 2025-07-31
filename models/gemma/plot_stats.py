#!/usr/bin/env python

# Plot stats from multi-case validation

from rainfall_rescue.utils.validate import (
    plot_metadata_fraction,
    plot_monthly_table_fraction,
    plot_totals_fraction,
    validate_case,
    merge_validated_cases,
)
from rainfall_rescue.utils.pairs import get_index_list
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
parser.add_argument(
    "--purpose",
    help="Training or test or neither",
    type=str,
    required=False,
    default="Test",
)
parser.add_argument(
    "--fake",
    help="Use fake cases instead of real",
    action="store_true",
    required=False,
    default=False,
)
args = parser.parse_args()


# Get the list of labels where there are extractions from the given model_id
def get_cases(model_id, purpose, fake=None):
    training = None
    if purpose.lower()[0:5] == "train":
        training = True
    elif purpose.lower() == "test":
        training = False
    labels = get_index_list(
        fake=fake,
        training=training,
    )

    opdir = f"{os.getenv('PDIR')}/extracted/{model_id}"
    if not os.path.exists(opdir):
        raise ValueError(f"Output directory {opdir} does not exist")
    extractions = os.listdir(opdir)
    if not extractions:
        raise ValueError(f"No extractions found for model {model_id}")
    extractions = [c[:-5] for c in extractions if c.endswith(".json")]

    # Filter labels to those that have extractions
    cases = [l for l in labels if l in extractions]
    return cases


# Find all the cases with extractions available from this model
cases = get_cases(args.model_id, args.purpose, args.fake)
if len(cases) == 0:
    raise ValueError(f"No cases found for model {args.model_id}")

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
fig.savefig("stats.webp")
