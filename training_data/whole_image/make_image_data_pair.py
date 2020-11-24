#!/usr/bin/env python

# Make a bare-bones imitation of a ten-year rainfall sheet
#  use random data.

import os

# All the details are in the tyrImage class
from tyrImage import tyrImage

# Specify where to put the output
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--docn", help="Document name", type=str, required=True)
parser.add_argument(
    "--opdir", help="output directory name", default=None, type=str, required=False
)
parser.add_argument("--xshift", help="pixels", type=int, default=None, required=False)
parser.add_argument("--yshift", help="pixels", type=int, default=None, required=False)
parser.add_argument(
    "--xscale", help="fraction", type=float, default=None, required=False
)
parser.add_argument(
    "--yscale", help="fraction", type=float, default=None, required=False
)
parser.add_argument(
    "--rotate", help="degrees clockwise", type=float, default=None, required=False
)
parser.add_argument(
    "--fontFamily",
    help="Arial|FreeSans|Nafees Web Naskh|...",
    type=str,
    default=None,
    required=False,
)
parser.add_argument(
    "--fontSize", help="points", type=float, default=None, required=False
)
parser.add_argument(
    "--fontStyle", help="normal|italic|slanted", type=str, default=None, required=False
)
parser.add_argument(
    "--fontWeight", help="normal|bold|light", type=str, default=None, required=False
)
parser.add_argument(
    "--jitterFontRotate",
    help="random degrees",
    type=float,
    default=None,
    required=False,
)
parser.add_argument(
    "--jitterFontSize", help="random points", type=float, default=None, required=False
)
parser.add_argument(
    "--jitterGridPoints",
    help="random fraction",
    type=float,
    default=None,
    required=False,
)
parser.add_argument(
    "--jitterLineWidth", help="random ?", type=float, default=None, required=False
)
args = parser.parse_args()
if args.opdir is None:
    args.opdir = ("%s/Robot_Rainfall_Rescue/training_data/") % os.getenv("SCRATCH")

# Pass all the image control options to image creation function
a = vars(args)
kwargs = {i: a[i] for i in a if i not in ["docn", "opdir"]}

ic = tyrImage(args.opdir, args.docn, **kwargs)
ic.makeImage()
ic.makeNumbers()
ic.dumpState()
