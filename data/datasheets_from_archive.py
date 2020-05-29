#!/usr/bin/env python

# Download the ten year rainfall sheets from the National Met Archive

import os
import subprocess

archive_dir = "https://digital.nmla.metoffice.gov.uk/download/file/"

local_dir = "%s/station_images/ten_year_rainfall/originals" % os.getenv("SCRATCH")

periods = ('1677-1886','1860-1869')

for period in periods:
    ldir = "%s/%s" % (local_dir,period)
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir)
    with open('urls/%s' % period,'r') as urlf:
        urls = [line.rstrip() for line in urlf]
    for url in urls:
        source = "%s/%s" % (archive_dir,url)
        target = "%s/%s.pdf"% (ldir,url)
        if os.path.exists(target):
            continue
        #print("wget %s -O %s" % (source, target))
        proc = subprocess.call("wget %s -O %s" % (source, target), shell=True)

