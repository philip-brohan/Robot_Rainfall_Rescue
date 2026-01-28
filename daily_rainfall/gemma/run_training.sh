#!/bin/bash

# Train Gemma3 4b against the Gemini3 training dataset

../../azure_tools/azure_run.py --experiment=DR_Gemma_2 --name=run_training --compute=A100x1 \
  -- ./train.py --model_id=google/gemma-3-4b-it --run_id=FineTuned/DR_Gemma \
  --epochs 5 --training_group=training

