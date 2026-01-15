#!/usr/bin/env python

# Granite debugging script - load model and data, prepare for training
import os
from PIL import Image
import argparse
from RR_utils.hf import HFlogin
from RR_utils.image import cut_image

HFlogin()  # Connect to Huggingface Hub - only needed for initial model weights download

from daily_rainfall.granite.test_collate_fn import make_example
from daily_rainfall.utils.load import (
    get_index_list,
    load_image,
    image_id_to_filename,
    image_id_to_transcription_filename,
    load_json,
)

import torch
from torch.utils.data import Dataset

from transformers import AutoProcessor

from peft import LoraConfig

# Text prompts - system and user
from daily_rainfall.smolvlm.prompts import s_prompt, u_prompt

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
    required=False,
    default=None,
)
parser.add_argument(
    "--patch_size",
    help="Image patch size (pixels)",
    type=int,
    required=False,
    default=None,
)
parser.add_argument(
    "--training_group",
    help="Transcription group to use for training",
    type=str,
    required=False,
    default="training",
)
parser.add_argument(
    "--validation_group",
    help="Transcription group to use for validation",
    type=str,
    required=False,
    default="validation",
)
clargs = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"


# Convert dataset to OAI messages
def format_data(sample):
    # Desired output is JSON ormated csv data
    assistant_message = sample[2]
    # Cut the image into squares
    img = sample[1]
    if clargs.patch_size is not None:
        blocks = cut_image(img, clargs.patch_size, overlap=0.1)
    else:
        blocks = [img]  # Use the whole image if no patch size is specified

    return {
        "messages": [
            # {
            #     "role": "system",
            #     "content": [{"type": "text", "text": s_prompt}],
            # },
            # {
            #     "role": "user",
            #     "content": [{"type": "image", "image": block} for block in blocks]
            #     + [
            #         {
            #             "type": "text",
            #             "text": s_prompt,
            #         },
            #         {
            #             "type": "text",
            #             "text": u_prompt,
            #         },
            #     ],
            # },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(assistant_message)}],
            },
        ],
    }


# Make a training dataset from the image/Gemini transcription pairs
class RRTrainingDataset(Dataset):
    def __init__(self, group=None):
        self.labels = get_index_list(group=group)
        self.group = group

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = load_image(image_id_to_filename(self.labels[idx]))
        json_data = load_json(
            image_id_to_transcription_filename(self.labels[idx], group=self.group)
        )
        return self.labels[idx], img, json_data


# train_dataset = RRTrainingDataset(group=clargs.training_group)
# # Convert dataset to OAI messages
# train_dataset = [format_data(sample) for sample in train_dataset]
test_dataset = RRTrainingDataset(group=clargs.validation_group)
test_dataset = [format_data(sample) for sample in test_dataset]


# Load model and tokenizer
processor = AutoProcessor.from_pretrained(clargs.model_id)
# model = AutoModelForImageTextToText.from_pretrained(clargs.model_id).to(device)

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
    % (os.getenv("PDIR"), clargs.run_id),  # directory to save and repository id
    num_train_epochs=clargs.epochs,  # number of training epochs
    per_device_train_batch_size=1,  # batch size per device during training
    gradient_accumulation_steps=4,  # number of steps before performing a backward/update pass
    gradient_checkpointing=True,  # use gradient checkpointing to save memory
    optim="adamw_torch_fused",  # use fused adamw optimizer
    logging_steps=5,  # log every 5 steps
    save_strategy="epoch",  # save checkpoint every epoch
    learning_rate=1e-4,  # 2e-4,  # learning rate, based on QLoRA paper
    bf16=False,  # use bfloat16 precision
    max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",  # use constant learning rate scheduler
    push_to_hub=False,  # push model to hub
    report_to="tensorboard",  # report metrics to tensorboard
    logging_dir="%s/%s/logs"
    % (os.getenv("PDIR"), clargs.run_id),  # directory to save logs
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },  # use reentrant checkpointing
    dataset_text_field="",  # need a dummy field for collator
    dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
)
sargs.remove_unused_columns = False  # important for collator


# Create a data collator to encode text and image pairs
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
    # print(examples)
    # sys.exit(0)
    for example in examples:
        print("1")
        print(example["messages"])
        print("2")
        # image_inputs = process_vision_info(example["messages"])
        # print(image_inputs)
        text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        print(text)
        texts.append(text.strip())
        # images.append(image_inputs)

    # Tokenize the texts and process the images
    sys.exit(0)
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
    labels = batch["input_ids"].clone()

    assistant_tokens = processor.tokenizer("<|assistant|>", return_tensors="pt")[
        "input_ids"
    ][0]
    eos_token = processor.tokenizer("<|end_of_text|>", return_tensors="pt")[
        "input_ids"
    ][0]

    for i in range(batch["input_ids"].shape[0]):
        apply_loss = False
        for j in range(batch["input_ids"].shape[1]):
            if not apply_loss:
                labels[i][j] = -100
            if (j >= len(assistant_tokens) + 1) and torch.all(
                batch["input_ids"][i][j + 1 - len(assistant_tokens) : j + 1]
                == assistant_tokens
            ):
                apply_loss = True
            if batch["input_ids"][i][j] == eos_token:
                apply_loss = False

    batch["labels"] = labels
    return batch


batch = collate_fn(test_dataset)

print("input_ids shape:", batch["input_ids"].shape)
print("attention_mask shape:", batch.get("attention_mask").shape)
print("labels shape:", batch["labels"].shape)
print("labels sample:")
print(batch["labels"])  # shows -100 masking before assistant tokens
