#!/usr/bin/env python

# Convert all the Ten-year rainfall images to tensors for ML model training

# Partition off 1/10 of them to be test data

# This script does not run the commands - it makes a list of commands
#  (in the file 'run.txt') which can be run in parallel.

import os

rootd = "%s/station_images/ten_year_rainfall/images" % os.getenv("SCRATCH")

# Function to check if the job is already done for this timepoint
def is_done(period, book, filen, purpose):
    op_file_name = "%s/%s/%s/%s/%s.tfd" % (rootd, purpose, period, book, filen,)
    if os.path.isfile(op_file_name):
        return True
    return False


f = open("run.sh", "w+")


count = 1
periods = os.listdir(rootd)
for period in periods:
    books = os.listdir("%s/%s" % (rootd, period))
    for book in books:
        files = os.listdir("%s/%s/%s" % (rootd, period, book))
        for filen in files:
            if filen == "0000.jpg":
                continue  # skip title pages
            if count % 10 == 0:
                if not is_done(period, book, filen, "test"):
                    cmd = (
                        './image_to_tensor.py --pern=%s --docn="%s" --filen="%s" --test\n'
                        % (period, book, filen)
                    )
                    f.write(cmd)
            else:
                if not is_done(period, book, filen, "training"):
                    cmd = (
                        './image_to_tensor.py --pern=%s --docn="%s" --filen="%s"\n'
                        % (period, book, filen)
                    )
                    f.write(cmd)
            count += 1

f.close()
