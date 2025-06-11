#!/usr/bin/env python

# Gemma model training script adapted for RRR
import os
import sys
from PIL import Image
import random
import argparse
from utils.hf import HFlogin

HFlogin()  # Connect to Huggingface Hub - only needed for initial model weights download

from rainfall_rescue.utils.pairs import get_index_list, load_pair

import torch
from torch.utils.data import Dataset

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

from peft import LoraConfig, PeftModel
from trl import SFTTrainer


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    help="Model ID",
    type=str,
    required=False,
    default="google/gemma-3-4b-it",
)
parser.add_argument(
    "--quantized",
    help="Use quantized model",
    action="store_true",
)
parser.add_argument(
    "--nmax",
    help="Maximum number of training cases to use",
    type=int,
    required=False,
    default=100,
)
parser.add_argument(
    "--run_id",
    help="Identifier for this training run",
    type=str,
    required=True,
    default=None,
)
parser.add_argument(
    "--year_fraction",
    help="Probability of asking for the start year",
    type=float,
    required=False,
    default=0.05,
)
parser.add_argument(
    "--total_fraction",
    help="Probability of asking for the annual total",
    type=float,
    required=False,
    default=0.1,
)
args = parser.parse_args()


# System message for the assistant
system_message = (
    "You are a climate scientist. Your task is to extract climate data from pages containing historical observations. "
    + "The pages you are working on are records of monthly rainfall from the UK Met Office. "
    + "Each page contains a data table with values for each calendar month, in each of ten years. "
    + "Each column is the data for one year, January at the top, December at the bottom. "
    + "Each row is the data for one calendar month, with January at the top and December at the bottom. "
    + "The first column is the month name. "
    + "At the bottom of the table is an extra row with totals for each year. "
    + "Sometimes some of the data values are missing, left blank. For missing values, return 'null'."
)

# User prompt that combines the user query and the schema
user_prompt = """What is the rainfall in the table for year <YEAR> and month <MONTH>?

<YEAR>
{year}
</YEAR>

<MONTH>
{month}
</MONTH>
"""

# Additional prompts for years, totals etc.
year_prompt = "What is the first year in the table?"

total_prompt = """What is the total rainfall for year <YEAR>?

<YEAR>
{year}
</YEAR>
"""


# Convert dataset to OAI messages
def format_data(sample):
    choice = random.choices(
        ["year", "total", "rainfall"],
        weights=[
            args.year_fraction,
            args.total_fraction,
            1 - args.year_fraction - args.total_fraction,
        ],
        k=1,
    )[0]
    csv = sample[2]
    year = random.sample(csv["Years"], 1)[0]
    year_idx = csv["Years"].index(year)
    month = random.sample(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        1,
    )[0]
    if choice == "rainfall":
        rainfall = csv[month][year_idx]
    else:
        rainfall = csv["Totals"][year_idx]
    try:
        rf = float(rainfall)
    except ValueError:
        rainfall = "null"
    if choice == "year":
        # Get the first year in the table
        year = csv["Years"][0]
        u_prompt = year_prompt
        assistant_message = year
    elif choice == "total":
        u_prompt = total_prompt.format(year=year)
        assistant_message = rainfall
    else:
        # Get the rainfall for the month
        u_prompt = user_prompt.format(
            year=year,
            month=month,
        )
        assistant_message = rainfall
    # cut down the image to a square
    img = sample[1]
    img = img.crop((0, 0, img.size[0], img.size[0]))
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img,
                    },
                    {"type": "text", "text": u_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_message}],
            },
        ],
    }


# Make a training dataset from the RR image/CSV pairs
class RRTrainingDataset(Dataset):
    def __init__(self, max_n=None):
        self.labels = get_index_list(max_n=max_n)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, csv = load_pair(self.labels[idx])
        return self.labels[idx], img, csv


dataset = RRTrainingDataset(max_n=args.nmax)
# Convert dataset to OAI messages
dataset = [format_data(sample) for sample in dataset]

print(dataset[77]["messages"])


# Define model init arguments
model_kwargs = dict(
    attn_implementation="eager",  # Use "flash_attention_2" when running on Ampere or newer GPU, and 'eager' for older GPUs
    torch_dtype=torch.bfloat16,  # What torch dtype to use, defaults to auto
    device_map="auto",  # Let torch decide how to load the model
)

# BitsAndBytesConfig int-4 config
if args.quantized:
    # Use quantization if specified
    model_kwargs["load_in_4bit"] = True
    model_kwargs["bnb_4bit_use_double_quant"] = True
    model_kwargs["bnb_4bit_quant_type"] = "nf4"
    model_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16
    model_kwargs["bnb_4bit_quant_storage"] = torch.bfloat16

# Load model and tokenizer
model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)
processor = AutoProcessor.from_pretrained(args.model_id)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)
from trl import SFTConfig

sargs = SFTConfig(
    output_dir="%s/%s"
    % (os.getenv("SCRATCH"), args.run_id),  # directory to save and repository id
    num_train_epochs=3,  # number of training epochs
    per_device_train_batch_size=1,  # batch size per device during training
    gradient_accumulation_steps=4,  # number of steps before performing a backward/update pass
    gradient_checkpointing=True,  # use gradient checkpointing to save memory
    optim="adamw_torch_fused",  # use fused adamw optimizer
    logging_steps=5,  # log every 5 steps
    save_strategy="epoch",  # save checkpoint every epoch
    learning_rate=2e-4,  # learning rate, based on QLoRA paper
    bf16=True,  # use bfloat16 precision
    max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",  # use constant learning rate scheduler
    push_to_hub=False,  # push model to hub
    report_to="tensorboard",  # report metrics to tensorboard
    logging_dir="%s/%s/logs"
    % (os.getenv("SCRATCH"), args.run_id),  # directory to save logs
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },  # use reentrant checkpointing
    dataset_text_field="",  # need a dummy field for collator
    dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
)
sargs.remove_unused_columns = False  # important for collator


# Create a data collator to encode text and image pairs
# Don't understand this bit - why can't the processor just operate on messages?


def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    # Iterate through each conversation
    for msg in messages:
        # Get content (ensure it's a list)
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        # Check each content element for images
        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                # Get the image and convert to RGB
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                image_inputs.append(image.convert("RGB"))
    return image_inputs


def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image_inputs = process_vision_info(example["messages"])
        text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text.strip())
        images.append(image_inputs)

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
    labels = batch["input_ids"].clone()

    # Mask image tokens
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]
    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels
    return batch


trainer = SFTTrainer(
    model=model,
    args=sargs,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)


# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# Save the final model
trainer.save_model()
# free the memory again
del model
del trainer
torch.cuda.empty_cache()


# We trained a LORA model, that's additional weights instead of of the base model.
# We need to merge the LORA weights with the base model weights to make a new base model

# Load Model base model
model = AutoModelForImageTextToText.from_pretrained(
    args.model_id, low_cpu_mem_usage=True
)

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, sargs.output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(
    "merged_model", safe_serialization=True, max_shard_size="2GB"
)
