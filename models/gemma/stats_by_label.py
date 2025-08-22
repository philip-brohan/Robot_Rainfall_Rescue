#!/usr/bin/env python

# Calculate stats from multi-case validation
# Identify the cases that do badly.

from rainfall_rescue.utils.validate import (
    validate_case,
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


# Validate all cases - we're only interested in overall quality
def get_summary(case):
    count_good = 0
    count_bad = 0
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
        for year in range(0, 10):
            if case[month][year]:
                count_good += 1
            else:
                count_bad += 1
    return count_good / 120


results = {}
for label in cases:
    results[label] = get_summary(validate_case(args.model_id, label))

# Print the results - going from largest to smallest
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for label, score in sorted_results:
    print(f"Case {label} has {score:.2%} good months")
