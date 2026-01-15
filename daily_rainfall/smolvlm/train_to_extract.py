#!/usr/bin/env python

# Batch extraction with SmolVLM over a specified transcription group (training/validation)

import os
import random
from PIL import Image
import argparse
from RR_utils.hf import HFlogin
from RR_utils.image import cut_image

HFlogin()  # Connect to Huggingface Hub - only needed for initial model weights download

from daily_rainfall.utils.load import (
    get_index_list,
    load_image,
    image_id_to_filename,
    image_id_to_transcription_filename,
    load_json,
)

import torch
from torch.utils.data import Dataset

from transformers import AutoProcessor, AutoModelForImageTextToText, TrainerCallback

from peft import LoraConfig, PeftModel
from trl import SFTTrainer


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    help="Model ID",
    type=str,
    required=False,
    default="HuggingFaceTB/SmolVLM-Instruct",
)
parser.add_argument(
    "--group",
    help="Transcription group to replicate
    type=str,
    required=False,
    default="validation
)
parser.add_argument(
    "--epoch",
    help="Training epoch to use
    type=int,
    required=False,
    default=None
)

clargs = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Text prompts - system and user
from daily_rainfall.smolvlm.prompts import s_prompt, u_prompt


# Convert dataset to OAI messages
def format_data(sample):
    # Desired output - json transcriptions
    assistant_message = sample[2]
    # Input image
    img = sample[1]
    blocks = [img]

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
                        "text": u_prompt,
                    },
                ],
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
        return self.labels[idx], img


dataset = RRTrainingDataset(group=clargs.roup)
# Convert dataset to OAI messages
dataset = [format_data(sample) for sample in train_dataset]

# Load model and tokenizer
processor = AutoProcessor.from_pretrained(
    clargs.model_id, size={"longest_edge": 5 * 384}
)
model = AutoModelForImageTextToText.from_pretrained(
    clargs.model_id, torch_dtype=torch.bfloat16
).to(device)

image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<image>")
]


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

    labels[labels == processor.tokenizer.pad_token_id] = (
        -100
    )  # Mask padding tokens in labels
    labels[labels == image_token_id] = -100  # Mask image token IDs in labels

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
        # Also compute train and eval loss for this epoch and log/save metrics
        try:
            tr = None
            if isinstance(kwargs, dict):
                tr = kwargs.get("trainer")
            if tr is None:
                tr = globals().get("trainer")

            def safe_eval_loss(trainer, dataset, name):
                if trainer is None or dataset is None:
                    print(f"Skipping {name} eval: trainer or dataset is None")
                    return None
                try:
                    res = trainer.evaluate(eval_dataset=dataset)
                    if not isinstance(res, dict):
                        print(f"Evaluation for {name} returned non-dict: {type(res)}")
                        return None
                    return res.get("eval_loss")
                except Exception as e:
                    print(f"Evaluation failed for {name}:", e)
                    return None

            # Prefer trainer attributes, fall back to globals
            train_ds = getattr(tr, "train_dataset", None) if tr is not None else None
            if train_ds is None:
                train_ds = globals().get("train_dataset")
            eval_ds = getattr(tr, "eval_dataset", None) if tr is not None else None
            if eval_ds is None:
                eval_ds = globals().get("test_dataset") or globals().get("eval_dataset")

            train_loss = safe_eval_loss(tr, train_ds, "train")
            eval_loss = safe_eval_loss(tr, eval_ds, "eval")

            epoch_num = int(getattr(state, "epoch", 0))
            t_str = (
                f"{train_loss:.4f}"
                if isinstance(train_loss, (int, float))
                else str(train_loss)
            )
            e_str = (
                f"{eval_loss:.4f}"
                if isinstance(eval_loss, (int, float))
                else str(eval_loss)
            )
            print(f"[epoch {epoch_num}] train_loss={t_str} eval_loss={e_str}")

            try:
                if tr is not None:
                    metrics = {}
                    if isinstance(train_loss, (int, float)):
                        metrics["train_loss"] = train_loss
                    if isinstance(eval_loss, (int, float)):
                        metrics["eval_loss"] = eval_loss
                    if metrics:
                        # log/save if available
                        if hasattr(tr, "log_metrics"):
                            tr.log_metrics("epoch", metrics)
                        if hasattr(tr, "save_metrics"):
                            tr.save_metrics("epoch", metrics)
            except Exception:
                pass
        except Exception as e:
            print("Failed to compute train/eval loss:", e)


trainer = SFTTrainer(
    model=model,
    args=sargs,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
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
