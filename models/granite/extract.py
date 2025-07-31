#!/usr/bin/env python

# Run the IBM Granite model and extract data from a RR image
# Output the extracted data as a JSON structure

import os
from RR_utils.hf import HFlogin
from RR_utils.image import cut_image
from rainfall_rescue.utils.pairs import get_index_list, load_pair, csv_to_json

HFlogin()

from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import argparse
import random

# Specify the model ID and image label
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    help="Model ID",
    type=str,
    required=False,
    default="ibm-granite/granite-vision-3.3-2b",
)
parser.add_argument(
    "--label",
    help="Image identifier",
    type=str,
    required=False,
    default=None,
)
parser.add_argument(
    "--fake",
    help="Use fake data - not real",
    action="store_true",
    required=False,
    default=False,
)
parser.add_argument(
    "--patch_size",
    help="Image patch size (pixels)",
    type=int,
    required=False,
    default=None,
)
args = parser.parse_args()

if args.label is None:
    args.label = random.choice(get_index_list(fake=args.fake))
    print(f"Label not specified. Using random label: {args.label}")

# Load the image and CSV data
img, csv = load_pair(args.label)
# Cut the image into blocks
if args.patch_size is not None:
    blocks = cut_image(img, args.patch_size, overlap=0.1)
else:
    blocks = [img]  # Use the whole image if no patch size is specified
print(f"Cut image into {len(blocks)} blocks of size {blocks[0].size}")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and processor
if os.path.exists(f"{os.getenv('PDIR')}/{args.model_id}"):
    model_dir = f"{os.getenv('PDIR')}/{args.model_id}"
    print(f"Loading model from local directory: {model_dir}")
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForVision2Seq.from_pretrained(model_dir).to(device)

else:
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForVision2Seq.from_pretrained(args.model_id).to(device)

# System prompt
s_prompt = (
    "You are a climate scientist. Your task is to extract climate data from pages containing historical observations. "
    + "The page you are working on is a record of monthly rainfall from one UK weather station. "
    + "At the top of each page is the name of a weather station, and the number of the station. "
    + "The station name will follow the words 'RAINFALL at' in the top centre of the page. "
    + "The station number will be in the top-right corner of the page. "
    + "The page contains a table with monthly rainfall data for ten years,  "
    + "The first row of the table gives the years. There will be 10 years. The first year will end in a 1, and the last year in a 0."
    + "The first column of the table is the month name, starting with January at the top and December at the bottom. "
    + "The bulk of the table gives values for each calendar month, in each of the ten years. "
    + "Each column is the data for one year, January at the top, December at the bottom. "
    + "There is sometimes an extra column on the right (after the last year) - ignore this column. "
    + "Each row is the data for one calendar month, with the first year on the left and the last year on the right. "
    + "At the bottom of the table is an extra row with totals for each year. "
    + "Sometimes some of the data values are missing, left blank. For missing values, return 'null'."
    + "Report the data in a JSON format. Don't include any other text. "
)

u_prompt = (
    "Output the data as a JSON object with the following structure:\n "
    + '{"Name":"<name>",'
    + '"Number":"<number>",'
    + '"Years":[<year1>,<year2>, ...],'
    + '"January":[<value1>,<value2>, ...],'
    + '"February":[<value1>,<value2>, ...],'
    + " And so on for months April to December"
    + '"Totals": [<total1>,<total2>,...]}'
)

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": img,
            },
            {
                "type": "text",
                "text": s_prompt,
            },
            {
                "type": "text",
                "text": u_prompt,
            },
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(device)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(
        **inputs, max_new_tokens=5000, do_sample=False, top_k=None, top_p=None
    )
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(csv_to_json(csv))
print(decoded)
Result = decoded[decoded.find("{") : decoded.rfind("}") + 1]

# Store the extracted values in a file
opfile = f"{os.getenv('PDIR')}/extracted/{args.model_id}/{args.label}.json"
os.makedirs(os.path.dirname(opfile), exist_ok=True)

with open(opfile, "w") as f:
    f.write(Result)
