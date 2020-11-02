#!/usr/bin/env python

# Make 10,000 fake images, each with associated data file
#  These images have a random perturbations in shift, scale, and rotation).

import os
import random

# Get the list of fonts to use
from fonts import fontNames
from fonts import fontScales

image_dir = "%s/ML_ten_year_rainfall/training_data/images" % os.getenv("SCRATCH")

f = open("run_mi.sh", "w+")

for idx in range(10000):
    fn = "%s/%04d.png" % (image_dir, idx)
    if os.path.exists(fn):
        continue
    fontFamily = random.choice(fontNames)
    fontStyle = random.choice(["normal", "italic", "oblique"])
    fontWeight = random.choice(["normal", "bold", "light"])
    fontSize = 11
    if fontFamily in fontScales:
        fontSize *= fontScales[fontFamily]
    xshift = random.randint(-30, 30)
    yshift = random.randint(-250, 250)
    xscale = random.normalvariate(1, 0.03)
    yscale = random.normalvariate(1, 0.03)
    rotate = random.normalvariate(0, 3)
    jitterFontRotate = random.normalvariate(0, 3)
    jitterFontSize = random.normalvariate(0, 1)
    jitterGridPoints = random.normalvariate(0, 0.001)
    jitterLineWidth = random.normalvariate(0, 0.25)
    f.write(
        (
            './make_image_data_pair.py --docn="%04d"'
            + " --xshift=%d --yshift=%d --xscale=%f --yscale=%f"
            + " --rotate=%f --jitterFontRotate=%f --jitterFontSize=%f"
            + " --jitterGridPoints=%f --jitterLineWidth=%f"
            + " --fontFamily='%s' --fontSize=%f"
            + " --fontStyle=%s --fontWeight=%s\n"
        )
        % (
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
