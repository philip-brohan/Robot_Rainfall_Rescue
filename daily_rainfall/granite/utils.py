# Granite-specific utility functions

import torch
from daily_rainfall.qwen.utils import process_vision_info


# Granite needs its own collator
class CollateFn:
    def __init__(self, processor, max_len=None):
        self.processor = processor
        self.max_len = max_len

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            image_inputs = process_vision_info(example["messages"])
            text = self.processor.apply_chat_template(
                example["messages"], add_generation_prompt=False, tokenize=False
            )
            texts.append(text.strip())
            images.append(image_inputs)

        # Tokenize the texts and process the images
        batch = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )

        # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
        labels = batch["input_ids"].clone()

        assistant_tokens = self.processor.tokenizer(
            "<|assistant|>", return_tensors="pt"
        )["input_ids"][0]
        eos_token = self.processor.tokenizer("<|end_of_text|>", return_tensors="pt")[
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
