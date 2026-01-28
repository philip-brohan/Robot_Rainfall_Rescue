#!/bin/bash

# Validate fine-tuned Granite against the Gemini3 validation dataset

for epoch in {1..5}; do
  for group in validation training; do
    echo "Validating group ${group} at epoch ${epoch}"
    ../../azure_tools/azure_run.py --experiment=Granite_2 --name=validate_e_${epoch}_g_${group} \
    --compute=H100x1  -- ./validate.py --base_model_id=ibm-granite/granite-vision-3.3-2b \
      --model_id=FineTuned/DR_Granite/merged_epoch_${epoch} --validation_group=${group}
  done
done