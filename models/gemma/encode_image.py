#!/usr/bin/env python

# What does encoding the image actually do?

import os
import sys
import random
from utils.hf import HFlogin
from rainfall_rescue.utils.pairs import get_index_list, load_pair

HFlogin()

from transformers import AutoProcessor
import torch
import numpy as np

import argparse

# Specify the model ID
parser = argparse.ArgumentParser()
parser.add_argument(
    "--label",
    help="Image identifier",
    type=str,
    required=False,
    default=None,
)
parser.add_argument(
    "--model_id",
    help="Model ID",
    type=str,
    required=False,
    default="google/gemma-3-4b-it",
)

args = parser.parse_args()
if args.label is None:
    args.label = random.choice(get_index_list())


# load the imagedata pair
img, csv = load_pair(args.label)


# The processor does the text+image<->token conversions

processor = AutoProcessor.from_pretrained(
    args.model_id,
    use_fast=True,
)

# here we're only going to do the image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to("cpu", dtype=torch.bfloat16)

for in_id in inputs.keys():
    print(in_id)
    print(inputs[in_id].shape)
    print("")

img2 = inputs["pixel_values"].to(dtype=torch.float32).numpy()
img_array = img2[0].transpose(1, 2, 0)  # Change shape to (896, 896, 3)
img_array = (
    (img_array * 255).clip(0, 255).astype(np.uint8)
)  # Scale and convert to uint8

# Use PIL to save the numpy array as an image
from PIL import Image

img2 = Image.fromarray(img_array)
img2.save("test.png")
