#!/usr/bin/env python

# Utility functions for training LLMs on daily rainfall

import os
from platform import processor
from PIL import Image
from peft import PeftModel
from RR_utils.image import cut_image
from pprint import pformat

from daily_rainfall.utils.load import (
    get_index_list,
    load_image,
    image_id_to_filename,
    image_id_to_transcription_filename,
    load_json,
)

import torch
from torch.utils.data import Dataset
from daily_rainfall.qwen.debug import pretty_batch_summary


# Make message in OpenAI API chat format
def format_data(
    label, image, s_prompt, u_prompt, target=None, patch_size=None, patch_overlap=0.1
):
    # Cut the image into patches?
    if patch_size is not None:
        blocks = cut_image(image, patch_size, overlap=patch_overlap)
    else:
        blocks = [image]  # Use the whole image if no patch size is specified

    msg = [
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
    ]
    if target is not None:
        msg.append(
            {
                "role": "assistant",  # Target is the expected output
                "content": [{"type": "text", "text": str(target)}],
            },
        )
    return {
        "messages": msg,
        "label": label,
    }


# Make a training dataset from the image/Gemini transcription pairs
class DRTrainingDataset(Dataset):
    def __init__(
        self,
        s_prompt,
        u_prompt,
        group="training",
        img_width=None,
        img_height=None,
        patch_size=None,
        patch_overlap=0.1,
    ):
        self.labels = get_index_list(group=group)
        self.group = group
        self.s_prompt = s_prompt
        self.u_prompt = u_prompt
        self.img_width = img_width
        self.img_height = img_height
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = load_image(image_id_to_filename(self.labels[idx]))
        if self.img_width is not None and self.img_height is not None:
            new_size = (self.img_width, self.img_height)
            img = img.resize(new_size, resample=Image.LANCZOS)
        else:
            if self.img_width is not None or self.img_height is not None:
                raise ValueError(
                    "Both img_width and img_height must be specified together."
                )
        if self.group is not None:
            target = load_json(
                image_id_to_transcription_filename(self.labels[idx], group=self.group)
            )
        else:
            target = None  # No target - we're doing inference
        return format_data(
            self.labels[idx],
            img,
            self.s_prompt,
            self.u_prompt,
            target=target,
            patch_size=self.patch_size,
            patch_overlap=self.patch_overlap,
        )


# Make an extraction dataset from a list of image IDs
class DRExtractDataset(Dataset):
    def __init__(
        self,
        s_prompt,
        u_prompt,
        image_list=[],
        img_width=None,
        img_height=None,
        patch_size=None,
        patch_overlap=0.1,
    ):
        self.image_list = image_list
        self.s_prompt = s_prompt
        self.u_prompt = u_prompt
        self.img_width = img_width
        self.img_height = img_height
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = load_image(image_id_to_filename(self.image_list[idx]))
        if self.img_width is not None and self.img_height is not None:
            new_size = (self.img_width, self.img_height)
            img = img.resize(new_size, resample=Image.LANCZOS)
        else:
            if self.img_width is not None or self.img_height is not None:
                raise ValueError(
                    "Both img_width and img_height must be specified together."
                )
        return format_data(
            self.image_list[idx],
            img,
            self.s_prompt,
            self.u_prompt,
            target=None,
            patch_size=self.patch_size,
            patch_overlap=self.patch_overlap,
        )


# Strip the images out of an OAI message and return stripped message and image list
# The Collate function needs this.
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


# We can use individual messages in OAI chat format for inference, but for training (and efficient inference) we need to collate into batches
# This is a class not a function so we can store processor, image_token_id, max_len etc. Instantiate it to get a callable collate function to pass to SFTTrainer.
class CollateFn:
    def __init__(self, processor, max_len=None):
        self.processor = processor
        self.max_len = max_len

    #  batch is a list of OAI {"messages": [...] } dicts - each containing a single message with images and text provided by a DRTrainingDataset.
    def __call__(self, batch):
        # 1) Build full chat text (includes assistant answer) - but replaces images with placeholder
        full_texts = [
            self.processor.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            for ex in batch
        ]

        # 2) Build prompt-only text (up to user turn; generation prompt on)
        if batch[0]["messages"][-1]["role"] == "assistant":
            prompt_texts = [
                self.processor.apply_chat_template(
                    ex["messages"][:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for ex in batch
            ]
        else:
            prompt_texts = full_texts

        # 3) Images
        # Build a per-example list of images (one list per example in batch)
        images = [process_vision_info(ex["messages"]) for ex in batch]

        # 4) Tokenize full inputs ONCE (text + images)
        enc = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )

        input_ids = enc["input_ids"]
        pad_id = self.processor.tokenizer.pad_token_id

        # 5) Compute prompt lengths with TEXT-ONLY tokenization (much cheaper than text+images)
        prompt_ids = self.processor.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=False,  # chat template already includes special tokens
        )["input_ids"]

        # Count non-pad tokens in prompt
        prompt_lens = (prompt_ids != pad_id).sum(dim=1)

        # 6) Labels: copy + mask prompt tokens + mask padding
        labels = input_ids.clone()
        bs, seqlen = labels.shape

        for i in range(bs):
            pl = int(prompt_lens[i].item())
            pl = min(pl, seqlen)
            labels[i, :pl] = -100

        # Mask padding positions too
        labels[labels == pad_id] = -100

        # If your processor produces pixel_values / image_grid_thw, keep them
        enc["labels"] = labels
        return enc


# Load a saved trained model state
def load_model_from_save(
    model_dir, base_model_id, model_class, processor_class, device
):
    # Try to load a full model from model_dir first (merged/full save)
    try:
        proc = processor_class.from_pretrained(model_dir)
    except Exception:
        proc = processor_class.from_pretrained(base_model_id)

    try:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = model_class.from_pretrained(model_dir, torch_dtype=dtype)
        model.to(device)
        return model, proc
    except Exception:
        # If model_dir only contains PEFT adapters, load base model then apply adapter
        model = model_class.from_pretrained(base_model_id)
        try:
            peft_model = PeftModel.from_pretrained(model, model_dir)
            peft_model.to(device)
            return peft_model, proc
        except Exception as e:
            raise RuntimeError(f"Failed to load model or PEFT from {model_dir}: {e}")


# Calculate validation statistics
def evaluate_loss(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if loss is None:
                continue
            # accumulate by batch size (labels contain -100 for ignored tokens)
            # compute effective token count for weighting
            label_mask = (batch["labels"] != -100).float()
            tokens = int(label_mask.sum().item())
            total_loss += float(loss.item()) * max(tokens, 1)
            total_tokens += tokens
    avg_loss = total_loss / total_tokens if total_tokens > 0 else None
    return avg_loss


# This doesn't work - model.generate() is failing (incorrect input)
# For the moment, use the extract method instead (this might be faster, if I can get it working).
def generate_output(
    model, dataloader, device, processor, max_new_tokens=5000, num_beams=1
):
    model.eval()
    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id
    results = []
    with torch.no_grad():
        for bidx, batch in enumerate(dataloader):
            try:
                print(
                    f"[generate_output] batch #{bidx} summary:\n{pformat(pretty_batch_summary(batch))}"
                )
            except Exception as e:
                print(f"[generate_output] failed to pretty-print batch summary: {e}")

            # Move tensors to device safely (leave non-tensors untouched)
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

            input_ids = batch["input_ids"]  # (B, T)
            labels = batch.get("labels", None)  # (B, T) with -100 for prompt/pad
            bs = input_ids.size(0)

            # Build batched prompt-only input_ids for generation:
            # mask out target tokens (labels != -100) by replacing them with pad_id
            gen_input_ids = input_ids.clone()
            if labels is not None:
                gen_input_ids[labels != -100] = pad_id
            attention_mask = (gen_input_ids != pad_id).long()

            # Find visual inputs (require both for this model)
            pv = batch.get("pixel_values", None)
            ig = batch.get("image_grid", None) or batch.get("image_grid_thw", None)

            if pv is None or ig is None:
                print(
                    "[generate_output] skipping batch: missing pixel_values or image_grid (both required)"
                )
                continue

            inputs = {
                "input_ids": gen_input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pv,
                # use the provided key name (image_grid or image_grid_thw)
                "image_grid": ig if "image_grid" in batch else None,
                "image_grid_thw": (
                    ig
                    if ("image_grid_thw" in batch and "image_grid" not in batch)
                    else None
                ),
            }
            # drop None-valued keys
            inputs = {k: v for k, v in inputs.items() if v is not None}

            try:
                print(
                    f"[generate_output] inputs summary:\n{pformat(pretty_batch_summary(inputs))}"
                )
            except Exception as e:
                print(f"[generate_output] failed to pretty-print inputs summary: {e}")

            # Single generate call for the whole batch
            gen_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, num_beams=num_beams
            )  # (B, S_gen)

            # Decode each example's generated ids and optionally the reference
            for i in range(gen_ids.size(0)):
                pred = tokenizer.decode(gen_ids[i], skip_special_tokens=True)

                ref = ""
                if labels is not None:
                    target_ids = input_ids[i][labels[i] != -100]
                    if target_ids.numel() > 0:
                        ref = tokenizer.decode(target_ids, skip_special_tokens=True)

                print(f"Prediction [{bidx}-{i}]: {pred}")
                results.append({"prediction": pred, "reference": ref})
    return results
