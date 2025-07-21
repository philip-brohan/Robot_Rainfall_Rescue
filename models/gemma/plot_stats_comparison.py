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
args = parser.parse_args()


# Extract the validation ststistics
def get_merged(model_id):
    opdir = f"{os.getenv('PDIR')}/extracted/{model_id}"
    cases = os.listdir(opdir)
    if not cases:
        raise ValueError(f"No cases found for model {model_id}")
    cases = [c[:-5] for c in cases if c.endswith(".json")]

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
fig.savefig(
    "c_%s_%s.webp"
    % (args.model_id_1.replace("/", "_"), args.model_id_2.replace("/", "_")),
)
