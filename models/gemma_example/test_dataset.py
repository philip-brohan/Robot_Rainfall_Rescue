#!/usr/bin/env python

# Explore dataset handling

from datasets import load_dataset
from PIL import Image
from utils.hf import HFlogin

HFlogin()  # Connect to Huggingface Hub

dataset = load_dataset("philschmid/amazon-product-descriptions-vlm", split="train")

# dataset[0] is the first item in the dataset - it's a dictionary with
#  keys ('image','Product Name','Selling Price', ...)

# Convert a dataset item to the OAI message format used by Gemma
# The inputs are a system prompt, a user prompt, and the gemma output ('assistant role')
# System prompt sets the background - same for all items
# User prompt is the question or task to be performed - in this case, we are
# asking for a description of the product based on its name,category, and image.
# Assistant message is the desired output - in this case, the product description.


def format_data(sample):
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
                        "type": "text",
                        "text": user_prompt.format(
                            product=sample["Product Name"],
                            category=sample["Category"],
                        ),
                    },
                    {
                        "type": "image",
                        "image": sample["image"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["description"]}],
            },
        ],
    }
