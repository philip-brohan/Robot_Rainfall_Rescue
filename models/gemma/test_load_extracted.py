#!/usr/bin/env python

# Extracted data from the models is sometimes bad JSON.
# This script tests the process for fixing it up and making it as useable as possible.

from rainfall_rescue.utils.validate import (
    make_null_json,
    jsonfix,
    quote_list_items,
    load_extracted,
)
import os
import json
import re
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
    "--label",
    help="Image identifier",
    type=str,
    required=True,
    default=None,
)
args = parser.parse_args()

# Get the raw model output
opfile = f"{os.getenv('PDIR')}/extracted/{args.model_id}/{args.label}.json"
if not os.path.exists(opfile):
    raise Exception(f"No extraction for {args.model_id} {args.label}")
with open(opfile, "r") as f:
    raw_j = f.read()
print(f"Raw data from {opfile}:\n{raw_j}\n")

# Get the fixed_up json
fixed_j = jsonfix(raw_j)
print(f"Fixed-up data:\n{fixed_j}\n")

# Get rid of any junk after the totals
last_match = None
for m in re.finditer(r'"Totals"\s*:\s*\[.*?\]', fixed_j, flags=re.DOTALL):
    last_match = m

trimmed = fixed_j[: last_match.end()] + "}" if last_match else fixed_j
print(f"Trimmed data:\n{trimmed}\n")

extracted = json.loads(trimmed)
print(f"Extracted data:\n{extracted}\n")
