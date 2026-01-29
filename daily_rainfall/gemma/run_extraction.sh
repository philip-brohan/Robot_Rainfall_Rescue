#!/bin/bash

# Generate daily precipitation from images with fine-tuned Gemma3 4B

../../azure_tools/azure_run.py --experiment=DR_Gemma_2 --name=run_extraction \
  --compute=A100x1 -- ./extract.py --base_model_id=google/gemma-3-4b-it \
  --model_id=FineTuned/DR_Gemma/merged_epoch_5 \
  --image_ids_file=../gemini3/for_validation.txt