#!/bin/bash

# Generate daily precipitation from images with fine-tuned Qwen3 4B

../../azure_tools/azure_run.py --experiment=DR_Qwen_2 --name=run_generation --compute=H100x1 -- ./extract.py \
 --base_model_id=Qwen/Qwen3-VL-4B-Instruct --model_id=FineTuned/DR_Qwen/merged_epoch_5 \
 --image_ids_file=../gemini3/sample_1000.txt