#!/bin/bash

# Generate daily precipitation from images with fine-tuned Ministral 3 3B

../../azure_tools/azure_run.py --experiment=DR_Mistral_3 --name=run_generation --compute=A100x1 -- ./extract.py \
 --base_model_id=mistralai/Ministral-3-3B-Instruct-2512 --model_id=FineTuned/DR_Mistral_3/merged_epoch_5 \
 --image_ids_file=../gemini3/for_validation.txt