#!/bin/bash

# Extract a set of images

# HuggingFaceTB/SmolVLM2-2.2B-Instruct

../../azure_tools/azure_run.py --experiment=DR_SmolVLM --name=extract_set --compute=A100x1 -- ./extract_multi.py --model_id=FineTuned/DR_SmolVLM/merged_epoch_5 --image_file=./test_validation_images.txt

