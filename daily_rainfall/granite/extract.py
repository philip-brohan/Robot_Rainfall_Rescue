#!/usr/bin/env python

# Load a saved trained Granite model state and generate a set of output predictions

import os
import argparse
import torch

from transformers import AutoProcessor, AutoModelForImageTextToText

# Text prompts - system and user
from daily_rainfall.smolvlm.prompts import s_prompt, u_prompt
from daily_rainfall.qwen.utils import (
    DRExtractDataset,
    load_model_from_save,
    image_id_to_transcription_filename,
)
from daily_rainfall.granite.config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    PATCH_SIZE,
)
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument(
    "--base_model_id",
    help="Base Model ID",
    type=str,
    required=False,
    default="ibm-granite/granite-vision-3.3-2b",
)
parser.add_argument(
    "--model_id",
    help="Model ID",
    type=str,
    required=False,
    default="FineTuned/DR_Granite/merged_epoch_5",
)
parser.add_argument(
    "--generation_group",
    help="Transcription group to use for generation",
    type=str,
    required=False,
    default="validation",
)
parser.add_argument(
    "--image_ids_file",
    help="File with image ids to process (one per line)",
    type=str,
    required=True,
)

clargs = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model, processor = load_model_from_save(
    model_dir=f"{os.getenv('PDIR')}/{clargs.model_id}",
    base_model_id=clargs.base_model_id,
    model_class=AutoModelForImageTextToText,
    processor_class=AutoProcessor,
    device=device,
)

# Load the list of image IDs to process
HERE = Path(__file__).resolve().parent  # Directory of this script
with open(f"{HERE}/{clargs.image_ids_file}", "r") as f:
    image_ids = [line.strip() for line in f.readlines() if line.strip()]

extract_dataset = DRExtractDataset(
    image_list=image_ids,
    s_prompt=s_prompt,
    u_prompt=u_prompt,
    img_height=IMAGE_HEIGHT,
    img_width=IMAGE_WIDTH,
    patch_size=PATCH_SIZE,
)

for message in extract_dataset:
    inputs = processor.apply_chat_template(
        message["messages"],
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
    print(f"Extraction from image {message['label']}:")
    print(decoded)
    Result = decoded[decoded.find("{") : decoded.rfind("}") + 1]
    # Store the extracted values in a file
    opfile = image_id_to_transcription_filename(message["label"], group=clargs.model_id)
    os.makedirs(os.path.dirname(opfile), exist_ok=True)

    with open(opfile, "w") as f:
        f.write(Result)
