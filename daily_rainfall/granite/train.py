#!/usr/bin/env python

# Granite model training script adapted for daily rainfall

import os
import argparse
from RR_utils.hf import HFlogin

HFlogin()  # Connect to Huggingface Hub - only needed for initial model weights download

import torch

from transformers import AutoProcessor, AutoModelForImageTextToText

# Text prompts - system and user
from daily_rainfall.smolvlm.prompts import s_prompt, u_prompt
from daily_rainfall.qwen.utils import DRTrainingDataset
from daily_rainfall.granite.utils import CollateFn
from daily_rainfall.granite.config import (
    set_SFTConfig,
    set_LoraConfig,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    PATCH_SIZE,
)
from daily_rainfall.qwen.trainer import CustomSFTTrainer, SaveStateCallback

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    help="Model ID",
    type=str,
    required=False,
    default="ibm-granite/granite-vision-3.3-2b",
)
parser.add_argument(
    "--epochs",
    help="Number of epochs to train",
    type=int,
    required=False,
    default=3,
)
parser.add_argument(
    "--run_id",
    help="Identifier for this training run",
    type=str,
    required=True,
    default=None,
)
parser.add_argument(
    "--training_group",
    help="Transcription group to use for training",
    type=str,
    required=False,
    default="training",
)
clargs = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = DRTrainingDataset(
    group=clargs.training_group,
    s_prompt=s_prompt,
    u_prompt=u_prompt,
    img_height=IMAGE_HEIGHT,
    img_width=IMAGE_WIDTH,
    patch_size=PATCH_SIZE,
)

# Load model and tokenizer
model = AutoModelForImageTextToText.from_pretrained(clargs.model_id).to(device)
processor = AutoProcessor.from_pretrained(clargs.model_id)
data_collator = CollateFn(processor=processor)

peft_config = set_LoraConfig()
sargs = set_SFTConfig(run_id=clargs.run_id, num_train_epochs=clargs.epochs)

data_collator = CollateFn(processor=processor)

trainer = CustomSFTTrainer(
    model=model,
    args=sargs,
    train_dataset=train_dataset,
    eval_dataset=None,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=data_collator,
    callbacks=[
        SaveStateCallback(out_dir=f"{os.getenv('PDIR')}/{clargs.run_id}"),
    ],
)

# Start training, the model will be saved to the output directory at the end of each epoch
try:
    trainer.train(
        resume_from_checkpoint=True
    )  # Auto restart if possible (job is likely to be preempted at some point)
except ValueError as e:
    trainer.train(resume_from_checkpoint=False)  # No checkpoint, start from scratch
