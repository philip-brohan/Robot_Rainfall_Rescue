#!/bin/bash

# Upload the validation transcriptions to Azure

../../azure_tools/azure_upload.py --remote=documents/Daily_Rainfall_UK/transcriptions/validation --local=$DOCS/Daily_Rainfall_UK/transcriptions/validation
