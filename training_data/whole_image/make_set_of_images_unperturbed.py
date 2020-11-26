#!/usr/bin/env python

# Make 10,000 fake images, each with associated data file

import os
import random

# Get the list of fonts to use
from fonts import fontNames
from fonts import fontScales

image_dir = "%s/Robot_Rainfall_Rescue/training_data/unperturbed/" % os.getenv("SCRATCH")

f = open("run_mi.sh", "w+")

for idx in range(10000):
    fn = "%s/images/%04d.png" % (image_dir, idx)
    if os.path.exists(fn):
        continue
    fontFamily = "Arial"
    fontStyle = "normal"
    fontWeight = "normal"
    fontSize = 10
    if fontFamily in fontScales:
        fontSize *= fontScales[fontFamily]
    xshift = 0
    yshift = 0
    xscale = 1
    yscale = 1
    rotate = 0
    jitterFontRotate = 0
    jitterFontSize = 0
    jitterGridPoints = 0
    jitterLineWidth = 0
    f.write(
        (
            './make_image_data_pair.py --opdir=%s --docn="%04d"'
            + " --xshift=%d --yshift=%d --xscale=%f --yscale=%f"
            + " --rotate=%f --jitterFontRotate=%f --jitterFontSize=%f"
            + " --jitterGridPoints=%f --jitterLineWidth=%f"
            + " --fontFamily='%s' --fontSize=%f"
            + " --fontStyle=%s --fontWeight=%s\n"
        )
        % (
            image_dir,
            idx,
            xshift,
            yshift,
            xscale,
            yscale,
            rotate,
            jitterFontRotate,
            jitterFontSize,
            jitterGridPoints,
            jitterLineWidth,
            fontFamily,
            fontSize,
            fontStyle,
            fontWeight,
        )
    )
