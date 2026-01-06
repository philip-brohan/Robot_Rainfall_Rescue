#!/usr/bin/env python

# List json filenames for all the Gemini-transcribed records.

import os

root = os.path.join(f"{os.getenv('DOCS')}/Daily_Rainfall_UK/transcriptions/Gemini3")
files = []
for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
    for fn in filenames:
        print(os.path.join(dirpath, fn)[79:])
