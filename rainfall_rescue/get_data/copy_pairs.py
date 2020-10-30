#!/usr/bin/env python

# Get all the error-free jpg::csv pairs from Ed's Rainfall Rescue dataset

import sys
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ind",
    help="Input directory",
    type=str,
    required=False,
    default="%s/rainfall-rescue-master/DATA/" % os.getenv("SCRATCH"),
)
parser.add_argument(
    "--outd",
    help="Output directory",
    type=str,
    required=False,
    default="%s/ML_ten_year_rainfall/from_Ed/" % os.getenv("SCRATCH"),
)
args = parser.parse_args()

for subd in ("images", "csvs"):
    sd = "%s/%s" % (args.outd, subd)
    if not os.path.isdir(sd):
        os.makedirs(sd)

# For each jpg, if there is a csv with the same name, copy both.
stations = os.listdir(args.ind)
for station in stations:
    sdir = "%s/%s" % (args.ind, station)
    if not os.path.isdir(sdir):
        continue
    fls = os.listdir(sdir)
    for fl in fls:
        if fl[-4:] == ".jpg":
            flc = "%s%s" % (fl[:-4], ".csv")
            if os.path.isfile("%s/%s" % (sdir, flc)):
                shutil.copy("%s/%s" % (sdir, flc), "%s/%s" % (args.outd, "csvs"))
                shutil.copy("%s/%s" % (sdir, fl), "%s/%s" % (args.outd, "images"))
