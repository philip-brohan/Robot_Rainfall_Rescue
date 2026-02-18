#!/bin/bash

# Validate fine-tuned Mistral3 3B against the Gemini3 validation dataset

for epoch in {5..5}; do
  for group in validation training; do
    echo "Validating group ${group} at epoch ${epoch}"
    ../../azure_tools/azure_run.py --experiment=DR_Mistral_3 --name=validate_e_${epoch}_g_${group} \
    --compute=A100x1 -- ./validate.py --base_model_id=mistralai/Ministral-3-3B-Instruct-2512 \
      --model_id=FineTuned/DR_Mistral_3/merged_epoch_${epoch} --validation_group=${group}
  done
done