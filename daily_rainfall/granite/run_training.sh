#!/bin/bash

# Train Granite against the Gemini3 training dataset

../../azure_tools/azure_run.py --experiment=DR_Granite --name=run_training --compute=A100x1 \
  -- ./train.py --model_id=ibm-granite/granite-vision-3.3-2b --run_id=FineTuned/DR_Granite \
  --epochs 10 --training_group=training

