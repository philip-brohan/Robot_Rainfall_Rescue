#!/bin/bash

# Test the new extraction with structured output
../../../azure_tools/azure_run.py --experiment=DR_structure_test --name=extract_test_1 --compute=T4x1 -- ./extract.py \
 --base_model_id=HuggingFaceTB/SmolVLM-Instruct \
 --image_id=DRain_1931-1940_RainNos_2286-2325_B047/DRain_1931-1940_RainNos_2286-2325_B047-0 \

# --base_model_id=HuggingFaceTB/SmolVLM-Instruct
# --base_model_id=HuggingFaceTB/SmolVLM-Instruct --model_id=FineTuned/DR_SmolVLM/merged_epoch_5
# --base_model_id=mistralai/Ministral-3-3B-Instruct-2512
# --base_model_id=ibm-granite/granite-vision-3.3-2b
# --base_model_id=google/gemma-3-4b-it --image_height=1971 --image_width=1200 --patch_size=600
