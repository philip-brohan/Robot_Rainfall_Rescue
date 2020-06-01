#!/usr/bin/env python

# Make indifidual page jpgs from the pdfs

import os
import glob
from pdf2image import convert_from_path
import tempfile

local_dir = "%s/station_images/ten_year_rainfall/" % os.getenv("SCRATCH")

periods = ("1890-1899", "1880-1889")

for period in periods:
    pdir = "%s/originals/%s" % (local_dir, period)
    jdir = "%s/images/%s" % (local_dir, period)
    pdfs = glob.glob("%s/*.pdf" % pdir)
    for pdf in pdfs:
        with tempfile.TemporaryDirectory() as tpath:
            pages = convert_from_path(pdf, 100, output_folder=tpath)
            tdir = "%s/%s" % (jdir, os.path.basename(pdf)[:-4])
            if not os.path.isdir(tdir):
                os.makedirs(tdir)
            for page_i in range(len(pages)):
                target = "%s/%04d.jpg" % (tdir, page_i)
                if os.path.exists(target):
                    continue
                pages[page_i].save(target, "JPEG")
