#!/bin/bash

# Train SmolVLM against the Gemini3 training dataset

../../azure_tools/azure_run.py --experiment=SmolVLM_2 --name=run_training --compute=A100x1 \
  -- ./train.py --model_id=HuggingFaceTB/SmolVLM-Instruct --run_id=FineTuned/DR_SmolVLM \
  --epochs 5 --training_group=training

