#!/bin/bash

# Validate fine-tuned Qwen3 4B against the Gemini3 validation dataset

for epoch in {1..2}; do
  for group in validation training; do
    echo "Validating group ${group} at epoch ${epoch}"
    ../../azure_tools/azure_run.py --experiment=DR_Qwen_2 --name=validate_e_${epoch}_g_${group} \
    --compute=A100x1  -- ./validate.py --base_model_id=Qwen/Qwen3-VL-4B-Instruct \
      --model_id=FineTuned/DR_Qwen/merged_epoch_${epoch} --validation_group=${group}
  done
done