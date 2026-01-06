#!/usr/bin/env python3

# Transcribe a daily-rainfall page using Gemini 3

import argparse
import os
import PIL.Image
import google.generativeai as genai
import typing_extensions as typing
from daily_rainfall.utils.load import (
    get_random_image_sample,
    load_image,
    get_json_name,
    save_json,
)

# You will need an API key. Get it from https://ai.google.dev/gemini-api/docs/api-key

# I keep my API key in the .gemini_api file in my home directory.
with open("%s/.gemini_api" % os.getenv("HOME"), "r") as file:
    api_key = file.read().strip()

# Default protocol is 'GRPC' - but that is blocked by the Office firewall.
#  Use 'REST' instead.
genai.configure(api_key=api_key, transport="rest")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--label",
    help="Image identifier",
    type=str,
    required=False,
    default=None,
)
args = parser.parse_args()

if args.label is None:
    image_name = get_random_image_sample(sample_size=1)[0]
    print(f"Label not specified. Using random image: {image_name}")

else:
    image_name = "%s/Daily_Rainfall_UK/jpgs_300dpi/%s" % (os.getenv("DOCS"), args.label)
    print(f"Loading image: {image_name}")

img = load_image(image_name)

# Don't bother with the metadata - we'll get it from the monthly rainfall


# Specify a structure for the daily observations
class Daily(typing.TypedDict):
    Day: int
    rainfall: str


class Monthly(typing.TypedDict):
    Month: str
    rainfall: list[Daily]
    total: str


class Annual(typing.TypedDict):
    Month: list[Monthly]


# Pick a random image
img = load_image(image_name)

# Pick an AI to use - this one is the latest as of 2025-01-29
# List available models
# models = genai.list_models()
# for model in models:
#     print(model.name)

model = genai.GenerativeModel("gemini-3-flash-preview")
# model = genai.GenerativeModel("gemini-3-pro-preview")
# model = genai.GenerativeModel("gemini-flash-latest")
# model = genai.GenerativeModel("gemini-2.5-pro")


# Get the daily observations from the image
result = model.generate_content(
    [
        img,
        "\n\n",
        "This image shows a table of daily rainfall observations for a single year. "
        "List the daily observations for months January to December. "
        "And the monthly totals at the bottom of each column. "
        "Be careful of missing data. Several days have missing data and "
        "These days will have an entry that is blank or has a dash '-'. "
        "Return the character '-' for missing data."
    ],
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json", response_schema=Annual
    ),
)

json_file_name = get_json_name(image_name, group="Gemini3")
save_json(result.text, json_file_name)

print(f"Saved transcription to {json_file_name}")
