#!/usr/bin/env python

# Get all the error-free jpg::csv pairs from Ed's Rainfall Rescue dataset

import sys
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ind",
    help="CSV directory",
    type=str,
    required=False,
    default="%s/rainfall-rescue-master/DATA/" % os.getenv("SCRATCH"),
)
parser.add_argument(
    "--imd",
    help="Image directory",
    type=str,
    required=False,
    default="%s/rainfall-rescue-master/IMAGES/" % os.getenv("SCRATCH"),
)
parser.add_argument(
    "--outd",
    help="Output directory",
    type=str,
    required=False,
    default="%s/Robot_Rainfall_Rescue/from_Ed/" % os.getenv("SCRATCH"),
)
args = parser.parse_args()

# Make the output directory (on SCRATCH - lots of Gb of images)
for subd in ("images", "csvs"):
    sd = "%s/%s" % (args.outd, subd)
    if not os.path.isdir(sd):
        os.makedirs(sd)

# Find all the images
images = {}
for dirpath, dirnames, filenames in os.walk(args.imd):
    for filename in filenames:
        if filename[-4:] != ".jpg":
            continue
        images[filename[:-4]] = dirpath + "/" + filename

# Find all the csvs
csvs = {}
for dirpath, dirnames, filenames in os.walk(args.ind):
    for filename in filenames:
        if filename[-4:] != ".csv":
            continue
        csvs[filename[:-4]] = dirpath + "/" + filename

# For each jpg, if there is a csv with the same name, copy both.
for image in images.keys():
    if image not in csvs:
        print("No csv for %s" % image)
        continue
    shutil.copy(images[image], "%s/%s" % (args.outd, "images"))
    shutil.copy(csvs[image], "%s/%s" % (args.outd, "csvs"))
