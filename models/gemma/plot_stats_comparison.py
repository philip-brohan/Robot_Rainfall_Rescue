#!/usr/bin/env python

# Plot stats from multi-case validation
# Compare between two models

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
    "--model_id_1",
    help="Model ID 1",
    type=str,
    required=False,
    default="google/gemma-3-4b-it",
)
parser.add_argument(
    "--model_id_2",
    help="Model ID 2",
    type=str,
    required=False,
    default="google/gemma-3-12b-it",
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


# Extract the validation ststistics
def get_merged(model_id):
    cases = get_cases(model_id, args.purpose, args.fake)
    if not cases:
        raise ValueError(f"No cases found for model {model_id}")

    # Validate all cases
    merged = None
    for label in cases:
        #    print(f"Validating case {label}")
        case = validate_case(model_id, label)
        if case is not None:
            merged = merge_validated_cases(merged, case)
    return merged


merged_1 = get_merged(args.model_id_1)
merged_2 = get_merged(args.model_id_2)

# Create the figure
fig = Figure(
    figsize=(7 * 2, 10),  # Width, Height (inches)
    dpi=100,
    facecolor=(0.95, 0.95, 0.95, 1),
    edgecolor=None,
    linewidth=0.0,
    frameon=True,
    subplotpars=None,
    tight_layout=None,
)
canvas = FigureCanvas(fig)

# Plot for model 1
ax_metadata_1 = fig.add_axes([0.05, 0.8, 0.45, 0.15])
plot_metadata_fraction(ax_metadata_1, merged_1)
ax_digitised_1 = fig.add_axes([0.05, 0.13, 0.45, 0.63])
plot_monthly_table_fraction(ax_digitised_1, merged_1)
ax_totals_1 = fig.add_axes([0.05, 0.05, 0.45, 0.07])
plot_totals_fraction(ax_totals_1, merged_1)

# Plot for model 2
ax_metadata_2 = fig.add_axes([0.05 + 0.45 + 0.03, 0.8, 0.45, 0.15])
plot_metadata_fraction(ax_metadata_2, merged_2, cmp=merged_1)
ax_digitised_2 = fig.add_axes([0.05 + 0.45 + 0.03, 0.13, 0.45, 0.63])
plot_monthly_table_fraction(ax_digitised_2, merged_2, cmp=merged_1, yticks=False)
ax_totals_2 = fig.add_axes([0.05 + 0.45 + 0.03, 0.05, 0.45, 0.07])
plot_totals_fraction(ax_totals_2, merged_2, cmp=merged_1)

# Render
fig.savefig("stats_comparison.webp")
