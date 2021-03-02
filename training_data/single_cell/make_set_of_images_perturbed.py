#!/usr/bin/env python

# Make 10,000 fake cell images, each with associated data file

import os
import random

# Get the list of fonts to use
from fonts import fontNames
from fonts import fontScales

image_dir = "%s/Robot_Rainfall_Rescue/training_data/cell/perturbed/" % os.getenv(
    "SCRATCH"
)

f = open("run_mi.sh", "w+")

for idx in range(10000):
    fn = "%s/images/%05d.png" % (image_dir, idx)
    if os.path.exists(fn):
        continue
    fontFamily = random.choice(fontNames)
    fontStyle = random.choice(["normal", "italic", "oblique"])
    fontWeight = random.choice(["normal", "bold", "light"])
    #fontFamily = "Arial"
    #fontStyle = "normal"
    #fontWeight = "normal"
    fontSize = 12
    if fontFamily in fontScales:
        fontSize *= fontScales[fontFamily]
    xshift = random.randint(-5, 5)
    yshift = random.randint(-5, 5)
    xscale = random.normalvariate(1, 0.03)
    yscale = random.normalvariate(1, 0.03)
    rotate = random.normalvariate(0, 3)
    jitterFontRotate = random.normalvariate(0, 3)
    jitterFontSize = random.normalvariate(0, 1)
    jitterGridPoints = random.normalvariate(0, 0.001)
    jitterLineWidth = random.normalvariate(0, 0.25)
    f.write(
        (
            './make_image_data_pair.py --opdir=%s --docn="%05d"'
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
