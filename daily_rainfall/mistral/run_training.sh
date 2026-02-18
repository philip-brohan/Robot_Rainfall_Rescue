#!/bin/bash

# Train Mistral against the Gemini3 training dataset

../../azure_tools/azure_run.py --experiment=DR_Mistral_3 --name=run_training --compute=A100x1 \
  -- ./train.py --model_id=mistralai/Ministral-3-3B-Instruct-2512 --run_id=FineTuned/DR_Mistral_3 \
  --epochs 5 --training_group=training

