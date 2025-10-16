#!/usr/bin/env python

from huggingface_hub import snapshot_download
import os

from RR_utils.hf import HFlogin

HFlogin()  # Connect to Huggingface Hub

# You could specify the tokenizers you want to download.
model_names = [
    # "Cosmos-Tokenizer-CI8x8",
    # "Cosmos-Tokenizer-CI16x16",
    # "Cosmos-Tokenizer-CV4x8x8",
    # "Cosmos-Tokenizer-CV8x8x8",
    # "Cosmos-Tokenizer-CV8x16x16",
    "Cosmos-Tokenizer-DI8x8",
    "Cosmos-Tokenizer-DI16x16",
    # "Cosmos-Tokenizer-DV4x8x8",
    # "Cosmos-Tokenizer-DV8x8x8",
    # "Cosmos-Tokenizer-DV8x16x16",
]
for model_name in model_names:
    hf_repo = "nvidia/" + model_name
    local_dir = "%s/pretrained_ckpts/%s" % (os.getenv("HF_HOME"), model_name)
    os.makedirs(local_dir, exist_ok=True)
    print(f"downloading {model_name} to {local_dir}...")
    snapshot_download(repo_id=hf_repo, local_dir=local_dir)
