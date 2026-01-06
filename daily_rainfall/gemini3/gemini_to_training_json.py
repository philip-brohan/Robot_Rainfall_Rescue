#! /usr/bin/env python

# Convert Gemini JSON output to training JSON format
# (The Gemini output is very verbose).

import os
import json
from daily_rainfall.utils.load import save_json, load_json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--record_id",
    help="Input Gemini JSON file",
    type=str,
    required=True,
)
args = parser.parse_args()

input_json = (
    f"{os.getenv('DOCS')}/Daily_Rainfall_UK/transcriptions/Gemini3/%s" % args.record_id
)
output_json = (
    f"{os.getenv('DOCS')}/Daily_Rainfall_UK/transcriptions/training/%s" % args.record_id
)

montharray = (
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
# Load the Gemini JSON file
gemini_data = load_json(input_json)
training = {}
for day in range(1, 32):
    day_key = f"Day {day}"
    training[day_key] = [None] * 12
    for month in gemini_data["Month"]:
        midx = month["Month"][:3]  # first 3 letters of month name
        gmd = month["rainfall"]
        for dy in gmd:
            if dy["Day"] == day:
                val = dy["rainfall"]
                if any(
                    c.isdigit() for c in str(val)
                ):  # Gemini value contains a decimal digit
                    try:
                        training[day_key][montharray.index(midx)] = val
                    except Exception as e:
                        print(f"Error processing day {day_key}, month {midx}: {e}")
                        print(training.keys())
                else:  # standardise all non-numeric values to 'null'
                    try:
                        training[day_key][montharray.index(midx)] = "null"
                    except Exception as e:
                        print(f"Error processing day {day_key}, month {midx}: {e}")
        val = month["total"]
        if "Totals" not in training:
            training["Totals"] = [None] * 12
        if any(c.isdigit() for c in str(val)):  # Gemini value contains a decimal digit
            training["Totals"][montharray.index(midx)] = val
        else:  # standardise all non-numeric values to 'null'
            training["Totals"][montharray.index(midx)] = "null"

os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, mode="w") as file:
    json.dump(training, file, indent=4)

