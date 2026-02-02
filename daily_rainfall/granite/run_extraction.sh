#!/bin/bash

# Generate daily precipitation from images with fine-tuned Granite

../../azure_tools/azure_run.py --experiment=DR_Granite_2 --name=run_extraction --compute=A100x1 -- ./extract.py \
  --base_model_id=ibm-granite/granite-vision-3.3-2b --model_id=FineTuned/DR_Granite/merged_epoch_5 \
  --image_ids_file=../gemini3/sample_1000.txt