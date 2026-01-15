#!/usr/bin/env python3
"""Run the image->text extractor over the entire selected group and save outputs.

Usage:
  python scripts/extract_group.py --model_id <model> --group training --batch_size 8
"""
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import torch

from RR_utils.hf import HFlogin

HFlogin()

from daily_rainfall.utils.load import (
    get_index_list,
    image_id_to_filename,
    load_image,
    get_json_name,
)
from daily_rainfall.smolvlm.prompts import s_prompt, u_prompt
from transformers import AutoProcessor, AutoModelForImageTextToText


def make_messages(img):
    return [
        {"role": "system", "content": [{"type": "text", "text": s_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": u_prompt},
            ],
        },
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="HuggingFaceTB/DR_SmolVLM")
    p.add_argument(
        "--group",
        default="training",
        help="which index group to run (training/validation)",
    )
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=2000)
    p.add_argument("--device", default=None)
    p.add_argument("--fp16", action="store_true")
    p.add_argument(
        "--epoch",
        type=int,
        required=False,
        help="Epoch number of the merged model to use",
        default=None,
    )
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # load model & processor (local copy preferred)
    model_dir = "%s/%s" % (os.getenv("PDIR"), args.model_id)
    if args.epoch is not None:
        model_dir = "%s/merged_epoch_%d" % (del_dir, args.epoch)
    print(f"Using model directory: {model_dir}")
    if os.path.exists(model_dir):
        print("Loading model from local dir:", model_dir)
        processor = AutoProcessor.from_pretrained(
            model_dir, size={"longest_edge": 5 * 384}
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=(torch.bfloat16 if not args.fp16 else None)
        )
    else:
        print("Loading model from hub:", args.model_id)
        processor = AutoProcessor.from_pretrained(
            args.model_id, size={"longest_edge": 5 * 384}
        )
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_id, dtype=(torch.bfloat16 if not args.fp16 else None)
        )

    model.to(device)
    model.eval()

    ids = get_index_list(group=args.group)
    print(f"Found {len(ids)} images in group '{args.group}'")

    def gen_batches(iterable, n):
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) >= n:
                yield batch
                batch = []
        if batch:
            yield batch

    for batch_ids in tqdm(list(gen_batches(ids, args.batch_size)), desc="batches"):
        imgs = []
        messages = []
        out_paths = []
        for image_id in batch_ids:
            fname = image_id_to_filename(image_id)
            img = load_image(fname).convert("RGB")
            imgs.append(img)
            messages.append(make_messages(img))
            out_paths.append(get_json_name(fname, group=args.model_id))

        # Build texts (include generation prompt) and process images
        texts = [
            processor.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False
            ).strip()
            for m in messages
        ]

        batch_enc = processor(
            text=texts, images=[[im] for im in imgs], return_tensors="pt", padding=True
        )
        input_ids = batch_enc["input_ids"].to(device)
        attention_mask = batch_enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # lengths per example (to slice off prompt after generation)
        input_lens = (
            attention_mask.sum(dim=1).tolist()
            if attention_mask is not None
            else [input_ids.shape[1]] * input_ids.shape[0]
        )

        enc_kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            enc_kwargs["attention_mask"] = attention_mask

        with torch.inference_mode():
            if args.fp16 and device.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    gen = model.generate(
                        **enc_kwargs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
            else:
                gen = model.generate(
                    **enc_kwargs, max_new_tokens=args.max_new_tokens, do_sample=False
                )

        print(gen.shape)
        print(gen)
        for i in range(gen.shape[0]):
            decoded = processor.decode(gen[i, :], skip_special_tokens=True)
            print(decoded)

        # For each item, strip the prompt tokens and decode
        gen = gen.cpu()
        for i in range(gen.shape[0]):
            inp_len = int(input_lens[i])
            gen_ids = gen[i, inp_len:]
            # remove potential leading pad tokens
            toks = [int(x) for x in gen_ids.tolist()]
            text = processor.tokenizer.decode(toks, skip_special_tokens=True)
            # crude JSON extraction between braces
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                result = text[start : end + 1]
            else:
                result = text

            outp = out_paths[i]
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            print(result)
            with open(outp, "w", encoding="utf-8") as f:
                f.write(result)

    print("Done")


if __name__ == "__main__":
    main()
