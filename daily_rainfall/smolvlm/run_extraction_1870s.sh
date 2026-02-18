#!/bin/bash

# Generate daily precipitation from images with fine-tuned SmolVLM
# Do the extraction in 10 streams for all 8162 1870s images

grep 1871 $DOCS/Daily_Rainfall_UK/all.txt > 1870s.txt
split -n l/10 -d --additional-suffix=.txt 1870s.txt 1870s_part_

for i in {00..09}; do
  echo "Running extraction for part $i"
  ../../azure_tools/azure_run.py --experiment=DR_SmolVLM_2_1870s --name=run_extraction_part_$i --compute=V100x1 -- ./extract.py \
    --base_model_id=HuggingFaceTB/SmolVLM-Instruct --model_id=FineTuned/DR_SmolVLM/merged_epoch_5 \
    --image_ids_file=1870s_part_${i}.txt
done
