#!/bin/bash

# Generate daily precipitation from images with fine-tuned Ministral 3 3B
# Do the extraction in 10 streams for all 8162 1870s images

for i in {00..09}; do
  echo "Running extraction for part $i"
  ../../azure_tools/azure_run.py --experiment=DR_Mistral_3_1870s --name=run_extraction_part_$i --compute=A100x1 -- ./extract.py \
    --base_model_id=mistralai/Ministral-3-3B-Instruct-2512 --model_id=FineTuned/DR_Mistral_3/merged_epoch_5 \
    --image_ids_file=../smolvlm/1870s_part_${i}.txt
done
