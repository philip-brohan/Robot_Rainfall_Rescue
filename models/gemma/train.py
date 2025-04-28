#!/usr/bin/env python

# Train a Gemma model to get the same outputs as RR

import os
from utils.hf import HFlogin
from rainfall_rescue.utils.pairs import get_index_list, load_pair

HFlogin()

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
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
)
args = parser.parse_args()


# Make a training dataset
class RRTrainingDataset(Dataset):
    def __init__(self):
        self.labels = get_index_list()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img,csv = load_pair(self.labels[idx])
        return img, csv

RR_ds = RRTrainingDataset()
RR_dataloader = DataLoader(RR_ds, batch_size=8, shuffle=True)


# Load the model and processor
model = Gemma3ForConditionalGeneration.from_pretrained(
    args.model_id, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(args.model_id)

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

p_prompt= "For each or the 10 years, output the rainfall for each of the 12 months, in the format {year: {'January': value, 'February': value, ...}}. "


Results = []
for q in Questions:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": s_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {
                    "type": "text",
                    "text": q,
                },
            ],
        },
    ]

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
    print(decoded)
    Results.append(decoded[decoded.find("{") : decoded.rfind("}") + 1])

# Store the extracted values in a file
opfile = f"{os.getenv('SCRATCH')}/Robot_Rainfall_Rescue/extracted/{args.model_id}/{args.label}.json"
os.makedirs(os.path.dirname(opfile), exist_ok=True)
with open(opfile, "w") as f:
    for result in Results:
        f.write(result + "\n")
