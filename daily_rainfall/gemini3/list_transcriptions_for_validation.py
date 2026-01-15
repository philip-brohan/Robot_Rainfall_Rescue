#!/usr/bin/env python

# Create a list of image ids where we have Gemini3 transcriptions that are not yet in the
# main transcription set.
# Note that many of these will be bad images (no data table) which we probably don't want.

import os

root = os.path.join(f"{os.getenv('DOCS')}/Daily_Rainfall_UK/transcriptions/Gemini3")
gemini = []
for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
    for fn in filenames:
        gemini.append(os.path.join(dirpath, fn)[79:])

root = os.path.join(f"{os.getenv('DOCS')}/Daily_Rainfall_UK/transcriptions/training")
training = []
for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
    for fn in filenames:
        training.append(os.path.join(dirpath, fn)[80:])
for id in gemini:
    if id not in training:
        print(id)
