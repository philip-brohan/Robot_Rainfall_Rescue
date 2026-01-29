#!/bin/bash

# Download all the transcriptions from Azure to local
# Skip the Training and Validation data from Gemini3

../../azure_tools/azure_download.py --remote=documents/Daily_Rainfall_UK/transcriptions/FineTuned --local=$DOCS/Daily_Rainfall_UK/transcriptions/FineTuned

