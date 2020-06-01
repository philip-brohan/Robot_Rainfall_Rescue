#!/usr/bin/env python

# Download the ten year rainfall sheets from the National Met Archive

import os
import subprocess

archive_dir = "https://digital.nmla.metoffice.gov.uk/download/file/"

local_dir = "%s/station_images/ten_year_rainfall/originals" % os.getenv("SCRATCH")

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

for period in periods:
    ldir = "%s/%s" % (local_dir, period)
    if not os.path.isdir(ldir):
        os.makedirs(ldir)
    with open("urls/%s" % period, "r") as urlf:
        urls = [line.rstrip() for line in urlf]
    for url in urls:
        source = "%s/%s" % (archive_dir, url)
        target = "%s/%s.pdf" % (ldir, url)
        if os.path.exists(target):
            continue
        # print("wget %s -O %s" % (source, target))
        proc = subprocess.call(
            "/usr/bin/wget --no-check-certificate %s -O %s" % (source, target),
            shell=True,
        )
