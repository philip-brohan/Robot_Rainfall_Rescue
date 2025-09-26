#!/usr/bin/env python

# Granite model training script adapted for RRR
import os
import random
from PIL import Image
import argparse
from RR_utils.hf import HFlogin
from RR_utils.image import cut_image

HFlogin()  # Connect to Huggingface Hub - only needed for initial model weights download

from rainfall_rescue.utils.pairs import get_index_list, load_pair, csv_to_json

import torch
from torch.utils.data import Dataset

from transformers import AutoProcessor, AutoModelForImageTextToText, TrainerCallback

from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# Text prompts - system and user
from models.smolvlm.prompts import s_prompt, u_prompt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    help="Model ID",
    type=str,
    required=False,
    default="ibm-granite/granite-vision-3.3-2b",
)
parser.add_argument(
    "--purpose",
    help="Training or test or neither",
    type=str,
    required=False,
    default="Training",
)
parser.add_argument(
    "--nmax",
    help="Maximum number of training cases to use",
    type=int,
    required=False,
    default=100,
)
parser.add_argument(
    "--fake",
    help="Use fake cases instead of real",
    action="store_true",
    required=False,
    default=False,
)
parser.add_argument(
    "--random_seed",
    help="Control the set of 'random'; choices",
    type=int,
    required=False,
    default=None,
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
    default=None,
)

clargs = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"


# Convert dataset to OAI messages
def format_data(sample):
    # Desired output is JSON ormated csv data
    assistant_message = csv_to_json(sample[2])
    # Cut the image into squares
    img = sample[1]
    if clargs.patch_size is not None:
        blocks = cut_image(img, clargs.patch_size, overlap=0.1)
    else:
        blocks = [img]  # Use the whole image if no patch size is specified

    return {
        "messages": [
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
                        "text": s_prompt,
                    },
                    {
                        "type": "text",
                        "text": u_prompt,
                    },
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
    def __init__(self, max_n=None, seed=None):
        if clargs.purpose.lower() == "test":
            self.labels = get_index_list(
                max_n=max_n, seed=seed, fake=clargs.fake, training=False
            )
        elif clargs.purpose[:5].lower() == "train":
            self.labels = get_index_list(
                max_n=max_n, seed=seed, fake=clargs.fake, training=True
            )
        else:
            self.labels = get_index_list(
                max_n=max_n, seed=seed, fake=clargs.fake, training=None
            )

        if clargs.random_seed is not None:
            torch.manual_seed(clargs.random_seed)
            random.seed(clargs.random_seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, csv = load_pair(self.labels[idx])
        return self.labels[idx], img, csv


dataset = RRTrainingDataset(max_n=clargs.nmax, seed=clargs.random_seed)
# Convert dataset to OAI messages
dataset = [format_data(sample) for sample in dataset]


# Load model and tokenizer
processor = AutoProcessor.from_pretrained(clargs.model_id)
model = AutoModelForImageTextToText.from_pretrained(clargs.model_id).to(device)

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
    bf16=True,  # use bfloat16 precision
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


# Define a callback to save a merged version of the model at the end of each epoch
class SaveMergedCallback(TrainerCallback):
    def __init__(self, base_model_id, out_dir):
        self.base_model_id = base_model_id
        self.out_dir = out_dir

    # Called at end of epoch
    def on_epoch_end(self, args, state, control, **kwargs):
        print("[SaveMergedCallback] on_epoch_end called")
        trainer.save_model()
        torch.cuda.empty_cache()
        model = AutoModelForImageTextToText.from_pretrained(
            clargs.model_id, low_cpu_mem_usage=True
        )
        # Merge LoRA and base model and save
        peft_model = PeftModel.from_pretrained(model, sargs.output_dir)
        merged_model = peft_model.merge_and_unload()
        merged_dir = os.path.join(
            sargs.output_dir, f"merged_epoch_{int(getattr(state,'epoch',0))}"
        )
        os.makedirs(merged_dir, exist_ok=True)
        merged_model.save_pretrained(
            merged_dir, safe_serialization=True, max_shard_size="2GB"
        )
        # Also need to save the processor and tokenizer to make a reusable model
        processor.save_pretrained(merged_dir)
        processor.tokenizer.save_pretrained(merged_dir)
        del merged_model
        del peft_model
        torch.cuda.empty_cache()


trainer = SFTTrainer(
    model=model,
    args=sargs,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
    callbacks=[SaveMergedCallback(clargs.model_id, sargs.output_dir)],
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
# model = AutoModelForImageTextToText.from_pretrained(args.model_id, low_cpu_mem_usage=True)

# # Merge LoRA and base model and save
# peft_model = PeftModel.from_pretrained(model, sargs.output_dir)
# merged_model = peft_model.merge_and_unload()
# merged_model.save_pretrained(
#     "merged_model", safe_serialization=True, max_shard_size="2GB"
# )
