#!/usr/bin/env python

# Run the SmolVLM model and extract data from a RR image
# Output the extracted data as a JSON structure

import os
from RR_utils.hf import HFlogin
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
    default="HuggingFaceM4/smolvlm-vision-2-seq",
)
parser.add_argument(
    "--image_count",
    help="No. of images to process",
    type=int,
    required=False,
    default=10,
)
parser.add_argument(
    "--fake",
    help="Use fake cases instead of real",
    action="store_true",
    required=False,
    default=False,
)
parser.add_argument(
    "--purpose",
    help="Training or test or neither",
    type=str,
    required=False,
    default="Test",
)
parser.add_argument(
    "--random_seed",
    help="Control the set of 'random'; choices",
    type=int,
    required=False,
    default=None,
)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and processor
if os.path.exists(f"{os.getenv('PDIR')}/{args.model_id}"):
    model_dir = f"{os.getenv('PDIR')}/{args.model_id}"
    print(f"Loading model from local directory: {model_dir}")
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(model_dir).to(device)

else:
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(args.model_id).to(device)


if args.random_seed is not None:
    print(f"Setting random seed to {args.random_seed}")
    random.seed(args.random_seed)

training = None
if args.purpose.lower() == "training":
    training = True
elif args.purpose.lower() == "test":
    training = False
labels = get_index_list(
    max_n=args.image_count,
    shuffle=True,
    seed=args.random_seed,
    fake=args.fake,
    training=training,
)
for label in labels:
    print(f"Using label: {label}")
    # Check to see if it's already been done (job might be restarting after being preempted)
    opfile = f"{os.getenv('PDIR')}/extracted/{args.model_id}/{label}.json"
    if os.path.exists(opfile):
        print(f"{label} already done - skipping")
        continue

    # Load the image and CSV data
    img, csv = load_pair(label)

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
    print(csv_to_json(csv))
    print(decoded)
    Result = decoded[decoded.find("{") : decoded.rfind("}") + 1]

    # Store the extracted values in a file
    opfile = f"{os.getenv('PDIR')}/extracted/{args.model_id}/{label}.json"
    os.makedirs(os.path.dirname(opfile), exist_ok=True)

    with open(opfile, "w") as f:
        f.write(Result)
