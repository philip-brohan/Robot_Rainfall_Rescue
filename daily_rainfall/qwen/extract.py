#!/usr/bin/env python

# Load a saved trained model state and generate a load of output predictions

import os
import argparse
import torch
from torch.utils.data import DataLoader

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Debugging
import sys
from pprint import pformat
from daily_rainfall.qwen.debug import pretty_batch_summary

# Text prompts - system and user
from daily_rainfall.smolvlm.prompts import s_prompt, u_prompt
from daily_rainfall.qwen.utils import (
    DRTrainingDataset,
    CollateFn,
    load_model_from_save,
    generate_output,
)
from daily_rainfall.qwen.config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--base_model_id",
    help="Base Model ID",
    type=str,
    required=False,
    default="Qwen/Qwen3-VL-4B-Instruct",
)
parser.add_argument(
    "--model_id",
    help="Model ID",
    type=str,
    required=False,
    default="FineTuned/DR_Qwen/merged_epoch_3",
)
parser.add_argument(
    "--generation_group",
    help="Transcription group to use for generation",
    type=str,
    required=False,
    default="validation",
)
parser.add_argument(
    "--patch_size",
    help="Image patch size (pixels)",
    type=int,
    required=False,
    default=None,
)
parser.add_argument(
    "--batch_size",
    help="Batch size",
    type=int,
    required=False,
    default=1,
)
clargs = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model, processor = load_model_from_save(
    model_dir=f"{os.getenv('PDIR')}/{clargs.model_id}",
    base_model_id=clargs.base_model_id,
    model_class=Qwen3VLForConditionalGeneration,
    processor_class=AutoProcessor,
    device=device,
)

eval_dataset = DRTrainingDataset(
    group=clargs.generation_group,
    s_prompt=s_prompt,
    u_prompt=u_prompt,
    img_height=IMAGE_HEIGHT,
    img_width=IMAGE_WIDTH,
    patch_size=clargs.patch_size,
)

for message in eval_dataset:
    inputs = processor.apply_chat_template(
        message["messages"],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    try:
        print(
            f"[eval_dataset] message summary:\n{pformat(pretty_batch_summary(inputs))}"
        )
    except Exception as e:
        print(f"[generate_output] failed to pretty-print inputs summary: {e}")

    with torch.inference_mode():
        generation = model.generate(
            **inputs, max_new_tokens=5000, do_sample=False, top_k=None, top_p=None
        )
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)
    Result = decoded[decoded.find("{") : decoded.rfind("}") + 1]
    sys.exit(0)
