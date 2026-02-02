# Utility functions for handling Daily Rainfall images and JSON transcriptions

import os
import csv
import json5 as json
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


def parse_station_metadata(path: str):
    """Parse a station CSV (RAPHOE-style) and return station_no, name, lat, long.

    Returns a dict: {'station_no': str or None, 'name': str or None,
    'lat': float or None, 'long': float or None}
    """
    station_no = None
    name = None
    lat = None
    long = None

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not any(cell.strip() for cell in row):
                continue
            first = row[0].strip()
            # station name: first non-empty row that isn't a keyword row
            if (
                name is None
                and first
                and not first.lower().startswith(
                    ("grid ref", "station no", "january", "february")
                )
            ):
                name = first
                # continue scanning for gridref / station no
                continue

            if first.lower().startswith("station no"):
                parts = [c.strip() for c in row if c.strip()]
                if len(parts) >= 2:
                    station_no = parts[1]
                continue

            if first.lower().startswith("grid ref"):
                # e.g. Grid ref,IC2117701042,Long,-7.670978,Lat,54.856858,Elevation,110,ft
                parts = [c.strip() for c in row if c.strip()]
                # parse key,value pairs
                for i in range(0, len(parts) - 1, 2):
                    key = parts[i].lower()
                    val = parts[i + 1]
                    if key.startswith("long"):
                        try:
                            long = float(val)
                        except Exception:
                            long = None
                    elif key.startswith("lat"):
                        try:
                            lat = float(val)
                        except Exception:
                            lat = None
                continue

    return {"station_no": station_no, "name": name, "lat": lat, "long": long}
