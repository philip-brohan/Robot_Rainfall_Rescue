#!/usr/bin/env python

# Experiment with data inputs

import os
import sys
from utils.hf import HFlogin
from rainfall_rescue.utils.pairs import get_index_list, load_pair

HFlogin()

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import argparse

# Specify the model ID
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    help="Model ID",
    type=str,
    required=False,
    default="google/gemma-3-4b-it",
)

args = parser.parse_args()


# Make a training dataset
class RRTrainingDataset(Dataset):
    def __init__(self, max_n=None):
        self.labels = get_index_list(max_n=max_n)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, csv = load_pair(self.labels[idx])
        return self.labels[idx], img, csv


RR_ds = RRTrainingDataset(max_n=10)
RR_dataloader = DataLoader(RR_ds, batch_size=2, shuffle=True)

label, img, rainfall = next(iter(RR_ds))

# print(rainfall)
# sys.exit(0)

# Load the model processor

processor = AutoProcessor.from_pretrained(
    args.model_id,
    use_fast=True,
)

# System prompt
s_prompt = (
    "You are a climate scientist. Your task is to extract climate data from pages containing historical observations. "
    + "The pages you are working on are records of monthly rainfall from the UK Met Office. "
    + "Each page contains a data table with values for each calendar month, in each of ten years. "
    + "Each column is the data for one year, January at the top, December at the bottom. "
    + "Each row is the data for one calendar month, with January at the top and December at the bottom. "
    + "The first column is the month name. "
    + "The last column is the average rainfall, for each year, in the calendar month"
    + "At the bottom of the table is an extra row with totals for each year. "
    + "Sometimes some of the data values are missing, left blank. For missing values, return 'null'."
    + "Report the data in a JSON format. Don't include any other text. "
)

p_prompt = "For each or the 10 years, output the rainfall for each of the 12 months, in the format {year: {'January': value, 'February': value, ...}}. "


messages = [
    # {
    #     "role": "system",
    #     "content": [{"type": "text", "text": s_prompt}],
    # },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            # {
            #     "type": "text",
            #     "text": p_prompt,
            # },
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
