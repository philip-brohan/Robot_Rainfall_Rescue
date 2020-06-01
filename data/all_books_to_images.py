#!/usr/bin/env python

# Make indifidual page jpgs from the pdfs

import os
import glob
from pdf2image import convert_from_path
import tempfile

local_dir = "%s/station_images/ten_year_rainfall/" % os.getenv("SCRATCH")

periods = (
    "1677-1886",
    "1860-1869",
    "1870-1879",
    "1880-1889",
    "1890-1899",
    "1900-1909",
    "1910-1919",
    "1920-1930",
    "1931-1940",
    "1941-1950",
    "1951-1960",
)

f = open("run_b2i.sh", "w+")

for period in periods:
    pdir = "%s/originals/%s" % (local_dir, period)
    pdfs = glob.glob("%s/*.pdf" % pdir)
    for pdf in pdfs:
        pdf = os.path.basename(pdf)
        f.write(
            (
                "conda activate ml_ten_year_rainfall; "
                + './book_to_images.py --pern=%s --docn="%s"\n'
            )
            % (period, pdf)
        )
