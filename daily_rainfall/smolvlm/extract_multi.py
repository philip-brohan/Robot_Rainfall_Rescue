#!/usr/bin/env python

# Run the SmolVLM model and extract data from a set of daily rainfall images
# Output the extracted data as a JSON structure

import os
from RR_utils.hf import HFlogin
from daily_rainfall.utils.load import (
    image_id_to_filename,
    image_id_to_transcription_filename,
    load_image,
)
from pathlib import Path

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
    default="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
)
parser.add_argument(
    "--image_file",
    help="File with image ids to process (one per line)",
    type=str,
    required=True,
)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the images to process
HERE = Path(__file__).resolve().parent  # Directory of this script
with open(f"{HERE}/{args.image_file}", "r") as f:
    image_ids = [line.strip() for line in f.readlines() if line.strip()]

# Load the model and processor
if os.path.exists(f"{os.getenv('PDIR')}/{args.model_id}"):
    model_dir = f"{os.getenv('PDIR')}/{args.model_id}"
    print(f"Loading model from local directory: {model_dir}")
    processor = AutoProcessor.from_pretrained(
        model_dir, size={"longest_edge": 15 * 384}
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",  # "flash_attention_2" if device == "cuda" else "eager",
    ).to(device)

else:
    processor = AutoProcessor.from_pretrained(
        args.model_id, size={"longest_edge": 15 * 384}
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",  # "flash_attention_2" if device == "cuda" else "eager",
    ).to(device)


for label in image_ids:
    print(f"Using image: {label}")
    # Check to see if it's already been done (job might be restarting after being preempted)
    opfile = image_id_to_transcription_filename(label, group=args.model_id)
    if os.path.exists(opfile):
        print(f"{label} already done - skipping")
        continue

    img = load_image(image_id_to_filename(label))

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": s_prompt}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img,
                },
                {
                    "type": "text",
                    "text": u_prompt,
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
            **inputs, max_new_tokens=5000, do_sample=False, top_k=None, top_p=None
        )
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)
    Result = decoded[decoded.find("{") : decoded.rfind("}") + 1]

    # Store the extracted values in a file
    opfile = image_id_to_transcription_filename(label, group=args.model_id)
    os.makedirs(os.path.dirname(opfile), exist_ok=True)

    with open(opfile, "w") as f:
        f.write(Result)
