#!/bin/bash

# Upload the training transcriptions to Azure

../../azure_tools/azure_upload.py --remote=documents/Daily_Rainfall_UK/transcriptions/training --local=$DOCS/Daily_Rainfall_UK/transcriptions/training

