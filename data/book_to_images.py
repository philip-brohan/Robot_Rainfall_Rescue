#!/usr/bin/env python

# Make indifidual page jpgs from one pdf

import os
from pdf2image import convert_from_path
import tempfile

local_dir = "%s/station_images/ten_year_rainfall/" % os.getenv("SCRATCH")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pern", help="Period", type=str, required=True)
parser.add_argument("--docn", help="Document name", type=str, required=True)
parser.add_argument(
    "--opdir", help="image directory name", default=None, type=str, required=False
)
args = parser.parse_args()
if args.opdir is None:
    args.opdir = ("%s/images/%s/%s") % (
        local_dir,
        args.pern,
        args.docn[:-4],  # strip the '.pdf'
    )
if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir)

pdf = "%s/originals/%s/%s" % (local_dir, args.pern, args.docn)

with tempfile.TemporaryDirectory() as tpath:
    pages = convert_from_path(pdf, 100, output_folder=tpath)
    for page_i in range(len(pages)):
        target = "%s/%04d.jpg" % (args.opdir, page_i)
        if os.path.exists(target):
            continue
        pages[page_i].save(target, "JPEG")
