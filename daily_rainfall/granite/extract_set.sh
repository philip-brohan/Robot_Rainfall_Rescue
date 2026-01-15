#!/bin/bash

# Extract a set of images

../../azure_tools/azure_run.py --experiment=DR_Granite --name=extract_set --compute=A100x1 -- ./extract_multi.py --model_id=FineTuned/DR_Granite/merged_epoch_5 --image_file=../gemini3/for_validation.txt

