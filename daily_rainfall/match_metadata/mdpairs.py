# Handle the RRdata and metadata

import os
import csv
import json
import re

root_dir = f"{os.getenv('PDIR')}/from_Ed/rainfall-rescue/DATA"


# find all the station .csv files
def find_csv_files():
    result = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".csv"):
                result.append(os.path.join(dirpath, filename))
    return result


# Load a csv file into a data structure (dictionary)
def load_station_csv(csv_path):
    result = {}
    with open(csv_path, mode="r", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        for index, row in enumerate(reader):
            row = ["null" if x == "" else x for x in row]
            if index == 0:
                result["Name"] = row[0]
            if index == 1:
                result["Longitude"] = row[3]
                result["Latitude"] = row[5]
            if index == 2:
                result["Number"] = row[1]
            if index == 4:
                result["Years"] = row[1:]
            if index >= 5 and index <= 16:  # Monthly data
                result[row[0].strip()] = row[1:]
            if index == 17:
                result["Totals"] = row[1:]
    return result
