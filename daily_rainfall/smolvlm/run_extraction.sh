#!/bin/bash

# Generate daily precipitation from images with fine-tuned SmolVLM

../../azure_tools/azure_run.py --experiment=DR_SmolVLM_2 --name=run_extraction --compute=V100x1 -- ./extract.py \
  --base_model_id=HuggingFaceTB/SmolVLM-Instruct --model_id=FineTuned/DR_SmolVLM/merged_epoch_5 \
  --image_ids_file=../gemini3/sample_1000.txt
