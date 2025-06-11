#!/usr/bin/env python

# Run one of the Gemma models and extract data from a RR image
# Output the extracted data as a JSON structure

import os
from utils.hf import HFlogin
from rainfall_rescue.utils.pairs import get_index_list, load_pair

HFlogin()

from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    AutoModelForImageTextToText,
)
from PIL import Image
import torch
import argparse
import random
import json

# Specify the model ID and image label
parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_id",
    help="Identifier for this training run",
    type=str,
    required=True,
    default="google/gemma-3-4b-it",
)
parser.add_argument(
    "--label",
    help="Image identifier",
    type=str,
    required=False,
    default=None,
)
args = parser.parse_args()

if args.label is None:
    args.label = random.choice(get_index_list())
    print(f"Label not specified. Using random label: {args.label}")

# Load the image and CSV data
img, csv = load_pair(args.label)
# Cut the image down to square
img1 = img.crop((0, 0, img.size[0], img.size[0]))
img2 = img.crop((0, img.size[1] - img.size[0], img.size[0], img.size[1]))

# Load the model and processor
if args.run_id.startswith("google/"):
    model = AutoModelForImageTextToText.from_pretrained(args.run_id)
    processor = AutoProcessor.from_pretrained(args.run_id)
else:
    model_dir = f"{os.getenv('SCRATCH')}/{args.run_id}"
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_dir, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_dir)

system_message = (
    "You are a climate scientist. Your task is to extract climate data from pages containing historical observations. "
    + "The pages you are working on are records of monthly rainfall from the UK Met Office. "
    + "Each page contains a data table with values for each calendar month, in each of ten years. "
    + "Each column is the data for one year, January at the top, December at the bottom. "
    + "Each row is the data for one calendar month, with January at the top and December at the bottom. "
    + "The first column is the month name. "
    + "At the bottom of the table is an extra row with totals for each year. "
    + "Sometimes some of the data values are missing, left blank. For missing values, return 'null'."
    + "Only output the values asked for. Don't include any other text."
)

# User prompt that combines the user query and the schema
user_prompt = """What is the rainfall in the table for year <YEAR> and month <MONTH>?

<YEAR>
{year}
</YEAR>

<MONTH>
{month}
</MONTH>
"""

# Additional prompts for years, totals etc.
year_prompt = "What is the first year in the table?"

total_prompt = """What is the total rainfall for year <YEAR>?

<YEAR>
{year}
</YEAR>
"""


# Function to extract a suingle data value
def extract_value(messages):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs, max_new_tokens=2000, do_sample=False, top_k=None, top_p=None
        )
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded


# First, get the start year
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": system_message}],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img1},
            {"type": "image", "image": img2},
            {
                "type": "text",
                "text": year_prompt,
            },
        ],
    },
]

start_year = extract_value(messages)

# Now the rainfall total for each year
years = [int(csv["Years"][0]) + i for i in range(10)]
totals = []
for year in years:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
                {
                    "type": "text",
                    "text": total_prompt.format(year=year),
                },
            ],
        },
    ]
    totals.append(extract_value(messages))

# Now the rainfall for each month in each year
rainfall = {}
for year in years:
    rainfall[year] = {}
    for month in [
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
    ]:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img1},
                    {"type": "image", "image": img2},
                    {
                        "type": "text",
                        "text": user_prompt.format(year=year, month=month),
                    },
                ],
            },
        ]
        rainfall[year][month] = extract_value(messages)


# Store the extracted values in a file
opfile = f"{os.getenv('SCRATCH')}/Robot_Rainfall_Rescue/extracted/{args.run_id}/{args.label}.json"
os.makedirs(os.path.dirname(opfile), exist_ok=True)
with open(opfile, "w") as f:
    json.dump(
        {"start_year": start_year, "totals": totals, "rainfall": rainfall}, f, indent=4
    )
