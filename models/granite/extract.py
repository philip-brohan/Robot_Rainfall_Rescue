#!/usr/bin/env python

# Run the IBM Granite model and extract data from a RR image
# Output the extracted data as a JSON structure

import os
from RR_utils.hf import HFlogin
from RR_utils.image import cut_image
from rainfall_rescue.utils.pairs import get_index_list, load_pair, csv_to_json

HFlogin()

from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import argparse
import random

# Text prompts - system and user
from models.smolvlm.prompts import s_prompt, u_prompt

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
    processor = AutoProcessor.from_pretrained("ibm-granite/granite-vision-3.3-2b")
    model = AutoModelForImageTextToText.from_pretrained(model_dir).to(device)

else:
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(args.model_id).to(device)


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
if len(Result) == 0:
    Result = decoded

# Store the extracted values in a file
opfile = f"{os.getenv('PDIR')}/extracted/{args.model_id}/{args.label}.json"
os.makedirs(os.path.dirname(opfile), exist_ok=True)

with open(opfile, "w") as f:
    f.write(Result)
