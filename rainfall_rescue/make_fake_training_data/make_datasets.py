#!/usr/bin/env python

# Make two datasets (training and test) of fake 10-year rainfall images

import os

# All the details are in the tyrImage class
from rainfall_rescue.make_fake_training_data.tyrImage.tyrImage import tyrImage

# Specify where to put the output
opdir = "%s/fake_training_data" % os.getenv("PDIR")
if not os.path.isdir(opdir):
    os.makedirs(opdir)

for i in range(1000):
    ic = tyrImage(opdir, "training_%03d" % i)

for i in range(100):
    ic = tyrImage(opdir, "test_%03d" % i)
