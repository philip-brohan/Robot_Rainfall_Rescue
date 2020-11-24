#!/usr/bin/env python

# Make all the individual frames for a movie

import os
import subprocess
import datetime

# Function to check if the job is already done for this timepoint
def is_done(epoch):
    op_file_name = (
        "%s/Robot_Rainfall_Rescue/models/ATB2_retuned/video/" + "%04d.png"
    ) % (os.getenv("SCRATCH"), int(epoch * 100))
    if os.path.isfile(op_file_name):
        return True
    return False


f = open("run.txt", "w+")

for epoch in range(100, 5020, 20):
    if is_done(epoch / 100):
        continue
    cmd = "./make_frame.py --epoch=%f \n" % (epoch / 100)
    f.write(cmd)
f.close()
