#!/usr/bin/env python

# Make 10,000 fake images, each with associated data file

import os

image_dir = "%s/ML_ten_year_rainfall/fakes/plain/images" % os.getenv("SCRATCH")

f = open("run_mi.sh", "w+")

for idx in range(10000):
    fn = "%s/%04d.png" % (image_dir, idx)
    if os.path.exists(fn):
        continue
    f.write(
        (
            "conda activate ml_ten_year_rainfall; "
            + './make_image_data_pair.py --docn="%04d"\n'
        )
        % idx
    )
