# Utility functions for handling Daily Rainfall images and JSON transcriptions

import os
import csv
import json
import random
import re

from PIL import Image


# Convert an image id to an image file name
def image_id_to_filename(image_id: str) -> str:
    filename = f"{os.getenv('DOCS')}/Daily_Rainfall_UK/jpgs_300dpi/{image_id}.jpg"
    return filename


# Convert an image id to a transcription file name
def image_id_to_transcription_filename(image_id: str, group="Test") -> str:
    filename = (
        f"{os.getenv('DOCS')}/Daily_Rainfall_UK/transcriptions/{group}/{image_id}.json"
    )
    return filename


# Get the indices of all images with a transcription in a given group
def get_index_list(group="Test"):
    root = os.path.join(f"{os.getenv('DOCS')}/Daily_Rainfall_UK/transcriptions/{group}")
    files = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for fn in filenames:
            image_id = f"{dirpath}/{fn}"
            image_id = image_id[len(root) + 1 :]
            if image_id.endswith(".json"):
                image_id = re.sub(r"\.json$", "", image_id)
                files.append(image_id)
    return files


# Get the image names (slow, there are 650,964 images)
def get_image_list():
    root = os.path.join(f"{os.getenv('DOCS')}/Daily_Rainfall_UK/jpgs_300dpi")
    files = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for fn in filenames:
            files.append(os.path.join(dirpath, fn))
    return files


# Load an Image from a file path
def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image {image_path} does not exist.")
    image = Image.open(image_path)
    return image


# Get a json file name from an image name
def get_json_name(image_name, group="Test"):
    if not image_name.endswith(".jpg"):
        raise ValueError(f"File {image_name} is not a JPEG image.")
    jfname = image_name.replace("jpgs_300dpi", "transcriptions/%s" % group).replace(
        ".jpg", ".json"
    )
    return jfname


# get an image name from a json file name
def get_image_name(json_name, group="Test"):
    if "transcriptions" not in json_name:
        raise ValueError(f"File {json_name} is not a transcription json file.")
    if not json_name.endswith(".json"):
        raise ValueError(f"File {json_name} is not a transcription json file.")
    imname = json_name.replace("transcriptions/%s" % group, "jpgs_300dpi").replace(
        ".json", ".jpg"
    )
    return imname


# Save a transcription json file
def save_json(json_data, json_path):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, mode="w") as file:
        json.dump(json.loads(json_data), file, indent=4)


# Load a transcription json file
def load_json(json_path):
    with open(json_path, mode="r") as file:
        json_data = json.load(file)
    return json_data


# Get a random sample of image file paths
def get_random_image_sample(sample_size=10):
    image_list = get_image_list()
    return random.sample(image_list, sample_size)
