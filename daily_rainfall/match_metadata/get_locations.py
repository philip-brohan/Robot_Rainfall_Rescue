#!/usr/bin/env python

# Go through all the rainfall rescue output files and lons

import os
import sys
import json5 as json
from daily_rainfall.match_metadata.mdpairs import find_csv_files, load_station_csv

# Get a list of all the RR pages = find_csv_files()
files = find_csv_files()

# Get a list of all the RR pages = find_csv_files()
files = find_csv_files()

meta = {}

# Get metadata
# Loop over all the station records
for p in files:
    csv = load_station_csv(p)

    try:
        station_name = csv["Name"]
        print(f"Processing station {station_name}")
    except KeyError:
        continue
    if station_name not in meta:
        meta[station_name] = {"number": "null", "latitude": "null", "longitude": "null"}
    try:
        if csv["Latitude"] != "null":
            meta[station_name]["latitude"] = float(csv["Latitude"])
    except Exception as e:
        print("Problem with Latitude in page:", p, "Error:", e)
    try:
        if csv["Longitude"] != "null":
            meta[station_name]["longitude"] = float(csv["Longitude"])
    except Exception as e:
        print("Problem with Longitude in page:", p, "Error:", e)
    try:
        if csv["Number"] != "null":
            meta[station_name]["number"] = csv["Number"]
    except Exception as e:
        print("Problem with Station Number in page:", p, "Error:", e)


# Save the metadata to a json file
outf = os.path.join(os.getenv("PDIR"), "station_metadata.json")
with open(outf, mode="w") as file:
    json.dump(meta, file, indent=4)
print(f"Saved metadata for {len(meta)} stations to {outf}")
