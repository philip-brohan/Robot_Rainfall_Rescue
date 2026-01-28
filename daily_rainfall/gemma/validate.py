#!/usr/bin/env python

# Load a saved trained model state and calculate validation loss
import os
import argparse
import torch
from torch.utils.data import DataLoader

from transformers import AutoProcessor, AutoModelForImageTextToText

# Text prompts - system and user
from daily_rainfall.smolvlm.prompts import s_prompt, u_prompt

from daily_rainfall.qwen.utils import (
    DRTrainingDataset,
    CollateFn,
    load_model_from_save,
    evaluate_loss,
)
from daily_rainfall.gemma.config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    PATCH_SIZE,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base_model_id",
    help="Base Model ID",
    type=str,
    required=False,
    default="google/gemma-3-4b-it",
)
parser.add_argument(
    "--model_id",
    help="Model ID",
    type=str,
    required=False,
    default="FineTuned/DR_Gemma/merged_epoch_5",
)
parser.add_argument(
    "--validation_group",
    help="Transcription group to use for validation",
    type=str,
    required=False,
    default="validation",
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
    model_class=AutoModelForImageTextToText,
    processor_class=AutoProcessor,
    device=device,
)

eval_dataset = DRTrainingDataset(
    group=clargs.validation_group,
    s_prompt=s_prompt,
    u_prompt=u_prompt,
    img_height=IMAGE_HEIGHT,
    img_width=IMAGE_WIDTH,
    patch_size=PATCH_SIZE,
)

loader = DataLoader(
    eval_dataset,
    batch_size=clargs.batch_size,
    collate_fn=CollateFn(processor=processor),
)

avg_loss = evaluate_loss(model, loader, device)
print(f"Model ID: {clargs.model_id}")
print(f"Loss for group {clargs.validation_group} = {avg_loss}")
