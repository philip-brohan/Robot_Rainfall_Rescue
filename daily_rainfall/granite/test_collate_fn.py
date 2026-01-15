#!/usr/bin/env python3
"""Quick test harness for the `collate_fn` used in granite/train.py

This file does NOT import `train.py` so it won't load the model or start training.
It defines a minimal compatible `processor` mock and runs a local `collate_fn`
implementation on synthetic examples to verify shapes and label masking.
"""
from PIL import Image
import torch


class MockTokenizer:
    def __call__(self, text, return_tensors=None):
        # Return small integer ids for special markers
        if text == "<|assistant|>":
            return {"input_ids": torch.tensor([[201, 202]])}
        if text == "<|end_of_text|>":
            return {"input_ids": torch.tensor([[999]])}
        # fallback: encode characters as small ints
        ids = [ord(c) % 100 for c in str(text)]
        return {"input_ids": torch.tensor([ids])}


class MockProcessor:
    def __init__(self):
        self.tokenizer = MockTokenizer()

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        # join text parts; messages are lists of {role, content:list}
        parts = []
        for m in messages:
            for c in m.get("content", []):
                # c is expected to have .text-like behavior (we use str())
                parts.append(str(c))
        return " ".join(parts)

    def __call__(self, text, images, return_tensors=None, padding=True):
        # Create padded input_ids tensor and attention_mask
        batch_size = len(text)
        seqs = []
        for i, t in enumerate(text):
            # produce a sequence that contains the assistant marker then some tokens then eos
            seq = [1, 2, 201, 202, 7 + i, 8 + i, 999]
            seqs.append(seq)
        maxlen = max(len(s) for s in seqs)
        input_ids = torch.zeros((batch_size, maxlen), dtype=torch.long)
        attn = torch.zeros((batch_size, maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, : len(s)] = 1
        return {"input_ids": input_ids, "attention_mask": attn}


def process_vision_info(messages):
    imgs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]
        for el in content:
            if isinstance(el, dict) and ("image" in el or el.get("type") == "image"):
                imgs.append(el.get("image"))
    return imgs


def _sanitize_messages(messages):
    out = []

    class _TextObj:
        __slots__ = ("type", "text")

        def __init__(self, text, t="text"):
            self.type = t
            self.text = text

        def __str__(self):
            return self.text

        def __add__(self, other):
            return str(self.text) + str(other)

        def __radd__(self, other):
            return str(other) + str(self.text)

    for m in messages:
        content = m.get("content", [])
        if not isinstance(content, list):
            content = [content]
        new_content = []
        for el in content:
            if isinstance(el, dict):
                if "text" in el:
                    new_content.append(_TextObj(el["text"], t=el.get("type", "text")))
                elif "image" in el or el.get("type") == "image":
                    new_content.append(_TextObj("", t="text"))
                else:
                    new_content.append(_TextObj(str(el), t="text"))
            else:
                new_content.append(_TextObj(str(el), t="text"))
        out.append({"role": m.get("role"), "content": new_content})
    return out


def collate_fn(examples, processor):
    texts = []
    images = []
    for example in examples:
        image_inputs = process_vision_info(example["messages"])
        safe_messages = _sanitize_messages(example["messages"])
        text = processor.apply_chat_template(safe_messages, add_generation_prompt=False, tokenize=False)
        texts.append(text.strip())
        images.append(image_inputs)

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()

    assistant_tokens = processor.tokenizer("<|assistant|>", return_tensors="pt")["input_ids"][0]
    eos_token = processor.tokenizer("<|end_of_text|>", return_tensors="pt")["input_ids"][0]

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


def make_example(img_text="hello", img_size=(8, 8)):
    # example shape: (id, image, assistant_message)
    img = Image.new("RGB", img_size, color=(255, 255, 255))
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {
            "role": "user",
            "content": [{"type": "image", "image": img}, {"type": "text", "text": img_text}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "assistant"}]},
    ]
    return {"messages": messages}


def main():
    processor = MockProcessor()
    examples = [make_example(f"text {i}") for i in range(3)]
    batch = collate_fn(examples, processor)

    print("input_ids shape:", batch["input_ids"].shape)
    print("attention_mask shape:", batch.get("attention_mask").shape)
    print("labels shape:", batch["labels"].shape)
    print("labels sample:")
    print(batch["labels"])  # shows -100 masking before assistant tokens


if __name__ == "__main__":
    main()
