#!/usr/bin/env python

# Run the IBM Granite model and extract data from a RR image
# Output the extracted data as a JSON structure

import os
from RR_utils.hf import HFlogin
from RR_utils.image import cut_image
from daily_rainfall.utils.load import (
    get_random_image_sample,
    load_image,
    get_json_name,
    save_json,
)

HFlogin()

from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import argparse
import random

# Text prompts - system and user
from daily_rainfall.smolvlm.prompts import s_prompt, u_prompt

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
args = parser.parse_args()

if args.label is None:
    args.label = get_random_image_sample(sample_size=1)[0]
    print(f"Label not specified. Using random label: {args.label}")
    image_name = args.label
else:
    image_name = "%s/%s" % (os.getenv("DOCS"), args.label)
    print(f"Loading image: {image_name}")

img = load_image(image_name)

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

print(decoded)
Result = decoded[decoded.find("{") : decoded.rfind("}") + 1]

json_file_name = get_json_name(image_name, group="Granite")
save_json(Result, json_file_name)

print(f"Saved transcription to {json_file_name}")
