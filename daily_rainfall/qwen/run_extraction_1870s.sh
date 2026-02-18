#!/bin/bash

# Generate daily precipitation from images with fine-tuned Qwen3 4B
# Do the extraction in 10 streams for all 8162 1870s images

for i in {00..09}; do
    ../../azure_tools/azure_run.py --experiment=DR_Qwen_2_1870s --name=run_extraction_part_$i --compute=A100x1 -- ./extract.py \
    --base_model_id=Qwen/Qwen3-VL-4B-Instruct --model_id=FineTuned/DR_Qwen/merged_epoch_5 \
    --image_ids_file=../smolvlm/1870s_part_${i}.txt
done