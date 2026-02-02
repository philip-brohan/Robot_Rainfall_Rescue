#!/usr/bin/env python

# Go through all the rainfall rescue output files and lons

import os
import sys
import json5 as json
from daily_rainfall.utils.load import parse_station_metadata

root = os.path.join(os.getenv("PDIR"), "../rainfall-rescue-master/DATA")

meta = {}
for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
    for fn in filenames:
        if not fn.endswith(".csv"):
            continue
        fpath = os.path.join(dirpath, fn)
        try:
            metadata = parse_station_metadata(fpath)
            if "station_no" not in metadata or metadata["station_no"] is None:
                continue
            number = metadata["station_no"]
            meta.setdefault(number, {})
            for value in ("name", "lat", "long"):
                if value not in metadata or metadata[value] is None:
                    continue
                meta[number][value] = metadata[value]
        except Exception as e:
            print(f"Failed to parse metadata from {fpath}: {e}")
        print(f"Processed {fpath}")

# Save the metadata to a json file
outf = os.path.join(os.getenv("PDIR"), "station_metadata.json")
with open(outf, mode="w") as file:
    json.dump(meta, file, indent=4)
print(f"Saved metadata for {len(meta)} stations to {outf}")
