# Utility functions for handling RR image/CSV pairs

import os
import csv
import json
import random
import re

from PIL import Image


# Get the image/csv names
def get_index_list(max_n=None, shuffle=True, seed=None, fake=False, training=None):
    if seed is not None:
        random.seed(seed)
    if fake:
        image_path = os.path.join(f"{os.getenv('PDIR')}/fake_training_data/images")
        result = [x[:-4] for x in os.listdir(image_path) if x.endswith(".jpg")]
    else:
        image_path = os.path.join(f"{os.getenv('PDIR')}/from_Ed/images")
        result = [x[:-4] for x in os.listdir(image_path) if x.endswith(".jpg")]
    if training is not None:
        if training:
            result = [x for x in result if x[-1] != "0"]
        else:
            result = [x for x in result if x[-1] == "0"]
    if shuffle:
        random.shuffle(result)
    if max_n is not None and len(result) > max_n:
        result = result[:max_n]
    return result


# Load a csv file into a data structure (dictionary)
def load_station_csv(csv_path):
    result = {}
    with open(csv_path, mode="r") as file:
        reader = csv.reader(file)
        for index, row in enumerate(reader):
            row = ["null" if x == "" else x for x in row]
            if index == 0:
                result["Name"] = row[0]
            if index == 2:
                result["Number"] = row[1]
            if index == 4:
                result["Years"] = row[1:11]
            if index >= 5 and index <= 16:  # Monthly data
                result[row[0]] = row[1:11]
            if index == 17:
                result["Totals"] = row[1:11]
    return result


# Convert the csv data to a json string
# To serve as target for the model
def csv_to_json(csv_data):
    """
    Convert CSV data to a JSON string.

    Args:
        csv_data (dict): The CSV data as a dictionary.

    Returns:
        str: JSON string representation of the CSV data.
    """
    csv_data["Number"] = str(csv_data["Number"])  # Easier if this is a string
    j = json.dumps(csv_data, separators=(",\n", ":"))
    # reformat to match LLM schema
    j = re.sub(
        r"\[([^\]]*)\]",
        lambda m: "[" + m.group(1).replace("\r", "").replace("\n", "") + "]",
        j,
        flags=re.DOTALL,
    )
    j = j.replace("{", "{\n")
    j = j.replace("}", "\n}")
    return j


# Load a pair of image and csv data
def load_pair(label):
    """
    Load a pair of image and CSV file based on the label.

    Args:
        label (str): The label of the image and CSV file.

    Returns:
        tuple: A tuple containing the image and the CSV file path.
    """
    if len(label) != 4:  # Real data
        image_path = os.path.join(f"{os.getenv('PDIR')}/from_Ed/images", f"{label}.jpg")
        csv_path = os.path.join(
            f"{os.getenv('PDIR')}/from_Ed/csvs",
            f"{label}.csv",
        )

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image {image_path} does not exist.")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV {csv_path} does not exist.")

        image = Image.open(image_path)
        csv = load_station_csv(csv_path)
    else:
        # Fake data
        image_path = os.path.join(
            f"{os.getenv('PDIR')}/fake_training_data/images", f"{label}.jpg"
        )
        csv_path = os.path.join(
            f"{os.getenv('PDIR')}/fake_training_data/csvs",
            f"{label}.csv",
        )

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image {image_path} does not exist.")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV {csv_path} does not exist.")

        image = Image.open(image_path)
        csv = load_fake_csv(csv_path)

    return image, csv


# Load a pair of image and csv data - from the fake training data
def load_fake_pair(label):
    image_path = os.path.join(
        f"{os.getenv('PDIR')}/fake_training_data/images", f"{label}.jpg"
    )
    csv_path = os.path.join(
        f"{os.getenv('PDIR')}/fake_training_data/csvs",
        f"{label}.csv",
    )

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image {image_path} does not exist.")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV {csv_path} does not exist.")

    image = Image.open(image_path)
    csv = load_fake_csv(csv_path)
    return image, csv


# Load a csv file into a data structure (dictionary)
def load_fake_csv(csv_path):
    result = {}
    with open(csv_path, mode="r") as file:
        contents = file.read()
        contents = contents.replace("'", '"')  # Json needs double quotes
        result = json.loads(contents)
    return result
