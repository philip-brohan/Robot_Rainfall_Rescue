#!/usr/bin/env python

# Run the Huggingface SmolVLM model and extract data from a Daily Rainfall image
# Output the extracted data as a JSON structure

import os
from RR_utils.hf import HFlogin
from daily_rainfall.utils.load import (
    image_id_to_filename,
    load_image,
    get_random_image_sample,
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
    default="HuggingFaceTB/SmolVLM-Instruct",
)
parser.add_argument(
    "--image",
    help="Image identifier",
    type=str,
    required=False,
    default=None,
)
args = parser.parse_args()

if args.image is None:
    image_name = get_random_image_sample(sample_size=1)[0]
    print(f"Image not specified. Using random image: {image_name}")

else:
    image_name = image_id_to_filename(args.image)
    print(f"Loading image: {image_name}")

img = load_image(image_name)

device = "cuda" if torch.cuda.is_available() else "cpu"

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

# Store the extracted values in a file
json_file_name = get_json_name(image_name, group=args.model_id)
os.makedirs(os.path.dirname(json_file_name), exist_ok=True)
with open(json_file_name, mode="w") as file:
    file.write(Result)


print(f"Saved transcription to {json_file_name}")
