#!/bin/bash

# Download the extractions made with SmolVLM on the Daily Rainfall validation set

../../azure_tools/azure_download.py --remote=documents/Daily_Rainfall_UK/transcriptions/FineTuned/DR_SmolVLM --local=$DOCS/Daily_Rainfall_UK/transcriptions/FineTuned/DR_SmolVLM/

