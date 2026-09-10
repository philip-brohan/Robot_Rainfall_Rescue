#!/usr/bin/env python

# Load a model and generate constrained JSON outputs from one or more images.

import os
import argparse
import torch

from pathlib import Path
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    LogitsProcessorList,
    # AutoModelForCausalLM,
)

import outlines
import json
from outlines.backends import get_json_schema_logits_processor


from daily_rainfall.structured_output.generic.prompts import s_prompt, u_prompt
from daily_rainfall.structured_output.generic.make_dataset import (
    DRExtractDataset,
    load_model_from_save,
)
from daily_rainfall.structured_output.generic.structure import RainfallRowTable


parser = argparse.ArgumentParser()
parser.add_argument(
    "--base_model_id",
    help="Base Model ID",
    type=str,
    required=True,
    default="HuggingFaceTB/SmolVLM-Instruct",
)
parser.add_argument(
    "--model_id",
    help="Model ID",
    type=str,
    required=False,
    default=None,
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
    required=False,
    default=None,
)
parser.add_argument(
    "--image_id",
    help="Single image id to process",
    type=str,
    required=False,
    default=None,
)
parser.add_argument(
    "--image_height",
    help="Height of the input images",
    type=int,
    required=False,
    default=None,
)
parser.add_argument(
    "--image_width",
    help="Width of the input images",
    type=int,
    required=False,
    default=None,
)
parser.add_argument(
    "--patch_size",
    help="Patch size for the input image",
    type=int,
    required=False,
    default=None,
)
clargs = parser.parse_args()

if clargs.image_ids_file is None and clargs.image_id is None:
    raise ValueError("Either --image_ids_file or --image_id must be provided.")
if clargs.image_ids_file is not None and clargs.image_id is not None:
    raise ValueError("Only one of --image_ids_file or --image_id can be provided.")

device = "cuda" if torch.cuda.is_available() else "cpu"

model, processor = load_model_from_save(
    model_id=clargs.model_id,
    base_model_id=clargs.base_model_id,
    model_class=AutoModelForImageTextToText,
    #    model_class=AutoModelForCausalLM,
    processor_class=AutoProcessor,
    device=device,
)

if not hasattr(processor, "tokenizer") or processor.tokenizer is None:
    raise RuntimeError(
        "Processor has no tokenizer; cannot initialize Outlines constrained decoding."
    )

try:
    outlines_model = outlines.from_transformers(model, processor)
    json_schema = json.dumps(RainfallRowTable.model_json_schema(by_alias=True))
    json_logits_processor = get_json_schema_logits_processor(
        backend_name=None,  # uses Outlines default JSON backend
        model=outlines_model,
        json_schema=json_schema,
    )
    logits_processors = LogitsProcessorList([json_logits_processor])
except Exception as e:
    raise RuntimeError(
        "Failed to initialize Outlines 1.2.12 JSON constrained decoding for this model/processor."
    ) from e

# Get the list of image IDs to process
if clargs.image_id is not None:
    image_ids = [clargs.image_id]
else:
    here = Path(__file__).resolve().parent
    with open(f"{here}/{clargs.image_ids_file}", "r") as f:
        image_ids = [line.strip() for line in f.readlines() if line.strip()]

extract_dataset = DRExtractDataset(
    image_list=image_ids,
    model_id=clargs.generation_group,
    s_prompt=s_prompt,
    u_prompt=u_prompt,
    img_height=clargs.image_height,
    img_width=clargs.image_width,
    patch_size=clargs.patch_size,
)

for message in extract_dataset:
    batch = processor.apply_chat_template(
        message["messages"],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Move tensors to device without corrupting integer ids.
    inputs = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                inputs[key] = value.to(model.device, dtype=torch.bfloat16)
            else:
                inputs[key] = value.to(model.device)
        else:
            inputs[key] = value

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=50000,
            do_sample=False,
            top_k=None,
            top_p=None,
            logits_processor=logits_processors,
        )
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True).strip()
    print(f"Decoded output for image {message['label']}:\n{decoded}\n")

    # Fail on invalid JSON or schema mismatch (no fallback).
    try:
        parsed = RainfallRowTable.model_validate_json(decoded)
        result_json = parsed.model_dump_json(indent=2, by_alias=True)
        print(f"Extraction from image {message['label']}:")
        print(result_json)
    except Exception as e:
        print(f"Failed to parse JSON for image {message['label']}: {e}")

# opfile = image_id_to_transcription_filename(
#     message["label"], group=clargs.generation_group
# )
# os.makedirs(os.path.dirname(opfile), exist_ok=True)

# with open(opfile, "w") as f:
#     f.write(result_json)
