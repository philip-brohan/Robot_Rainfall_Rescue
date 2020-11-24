#!/usr/bin/env python

# Assemble examples of the training data

import os
import pickle
from shutil import copyfile
from pprint import pprint

# Single example of unperturbed image
copyfile(
    "%s/Robot_Rainfall_Rescue/training_data_unperturbed/images/0000.png"
    % os.getenv("SCRATCH"),
    "unperturbed/0000.png",
)
# Data file for same image
with open("%s/Robot_Rainfall_Rescue/training_data_unperturbed/numbers/0000.pkl"
    % os.getenv("SCRATCH"),'rb') as pkf:
   dta = pickle.load(pkf)

with open('unperturbed/0000.py','w') as txf:
   pprint(dta,stream=txf)

# Four examples of perturbed images
for pfile in ('0000','0001','0002','0003'):
   copyfile(
       "%s/Robot_Rainfall_Rescue/training_data/images/%s.png"
       % (os.getenv("SCRATCH"),pfile),
       "perturbed/%s.png" % pfile
   )
