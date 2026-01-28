#!/bin/bash

# Validate fine-tuned Gemma3 4B against the Gemini3 validation dataset

../../azure_tools/azure_run.py --experiment=DR_Gemma_2 --name=validation_single --compute=A100x1 \
  -- ./validate.py --base_model_id=google/gemma-3-4b-it \
  --model_id=FineTuned/DR_Gemma/merged_epoch_5 --validation_group=validation