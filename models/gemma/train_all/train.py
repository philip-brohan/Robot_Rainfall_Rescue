#!/usr/bin/env python

# Gemma model training script adapted for RRR
import os
from PIL import Image
import argparse
from RR_utils.hf import HFlogin
from RR_utils.image import cut_image

HFlogin()  # Connect to Huggingface Hub - only needed for initial model weights download

from rainfall_rescue.utils.pairs import get_index_list, load_pair, csv_to_json

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)

from peft import LoraConfig, PeftModel
from trl import SFTTrainer


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    help="Model ID",
    type=str,
    required=False,
    default="google/gemma-3-12b-it",
)
parser.add_argument(
    "--no_quantize",
    help="Don't quantize the model",
    action="store_true",
    required=False,
    default=False,
)
parser.add_argument(
    "--nmax",
    help="Maximum number of training cases to use",
    type=int,
    required=False,
    default=100,
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
    "--patch_size",
    help="Image patch size (pixels)",
    type=int,
    required=False,
    default=600,
)

args = parser.parse_args()


# System message for the assistant
system_message = (
    "You are a climate scientist. Your task is to extract climate data from pages containing historical observations. "
    + "The page you are working on is a record of monthly rainfall from one UK weather station. "
    + "At the top of each page is the name of a weather station, and the number of the station. "
    + "The station name will follow the words 'RAINFALL at' in the top centre of the page. "
    + "The station number will be in the top-right corner of the page. "
    + "The page contains a table with monthly rainfall data for ten years,  "
    + "The first row of the table gives the years. There will be 10 years. The first year will end in a 1, and the last year in a 0."
    + "The first column of the table is the month name, starting with January at the top and December at the bottom. "
    + "The bulk of the table gives values for each calendar month, in each of the ten years. "
    + "Each column is the data for one year, January at the top, December at the bottom. "
    + "There is sometimes an extra column on the right (after the last year) - ignore this column. "
    + "Each row is the data for one calendar month, with the first year on the left and the last year on the right. "
    + "At the bottom of the table is an extra row with totals for each year. "
    + "Sometimes some of the data values are missing, left blank. For missing values, return 'null'."
    + "Report the data in a JSON format. Don't include any other text. "
)

# User prompt that combines the user query and the schema
user_prompt = (
    "Output the data as a JSON object with the following structure:\n "
    + '{"Name":"<name>",'
    + '"Number":"<number>",'
    + '"Years":[<year1>,<year2>, ...],'
    + '"January":[<value1>,<value2>, ...],'
    + '"February":[<value1>,<value2>, ...],'
    + " And so on for months April to December"
    + '"Totals": [<total1>,<total2>, ...]}'
)


# Convert dataset to OAI messages
def format_data(sample):
    # Desired output is JSON ormated csv data
    assistant_message = csv_to_json(sample[2])
    # Cut the image into squares
    img = sample[1]
    if args.patch_size is not None:
        blocks = cut_image(img, args.patch_size, overlap=0.1)
    else:
        blocks = [img]  # Use the whole image if no patch size is specified

    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [{"type": "image", "image": block} for block in blocks]
                + [
                    {
                        "type": "text",
                        "text": user_prompt,
                    }
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
if not args.no_quantize:
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
        bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
    )

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
    % (os.getenv("PDIR"), args.run_id),  # directory to save and repository id
    num_train_epochs=args.epochs,  # number of training epochs
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
    % (os.getenv("PRIR"), args.run_id),  # directory to save logs
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
try:
    trainer.train(
        resume_from_checkpoint=True
    )  # Auto restart if possible (job is likely to be preempted at some point)
except ValueError as e:
    trainer.train(resume_from_checkpoint=False)  # No checkpoint, start from scratch

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
