#!/bin/bash

# Validate fine-tuned SmolVLM against the Gemini3 validation dataset

for epoch in {3..5}; do
  for group in validation training; do
    echo "Validating group ${group} at epoch ${epoch}"
    ../../azure_tools/azure_run.py --experiment=SmolVLM --name=validate_e_${epoch}_g_${group} \
    --compute=A100x1  -- ./validate.py --base_model_id=HuggingFaceTB/SmolVLM-Instruct \
      --model_id=FineTuned/SmolVLM_2/merged_epoch_${epoch} --validation_group=${group}
  done
done