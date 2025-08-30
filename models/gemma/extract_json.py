#!/usr/bin/env python

# Run one of the Gemma models and extract data from a RR image
# Output the extracted data as a JSON structure

import os
from RR_utils.hf import HFlogin
from RR_utils.image import cut_image
from rainfall_rescue.utils.pairs import get_index_list, load_pair, csv_to_json

HFlogin()

from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    BitsAndBytesConfig,
)
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
    default="google/gemma-3-4b-it",
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
    default=600,
)
parser.add_argument(
    "--quantize",
    help="Quantize the model",
    action="store_true",
    required=False,
    default=False,
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

model_kwargs = dict(
    attn_implementation="eager",  # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch.bfloat16,  # What torch dtype to use, defaults to auto
    device_map="auto",  # Let torch decide how to load the model
)

# BitsAndBytesConfig int-4 config
if args.quantize:
    print("Using quantization for the model")
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
        bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
    )


# Load the model and processor
if os.path.exists(f"{os.getenv('PDIR')}/{args.model_id}"):
    model_dir = f"{os.getenv('PDIR')}/{args.model_id}"
    print(f"Loading model from local directory: {model_dir}")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_dir, **model_kwargs
    ).eval()
    processor = AutoProcessor.from_pretrained(model_dir)

else:
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_id, **model_kwargs
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_id)


messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": s_prompt}],
    },
    {
        "role": "user",
        "content": [{"type": "image", "image": block} for block in blocks]
        + [
            {
                "type": "text",
                "text": u_prompt,
            }
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
