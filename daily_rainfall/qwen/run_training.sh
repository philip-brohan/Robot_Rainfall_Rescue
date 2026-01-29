#!/bin/bash

# Train Qwen3 4B against the Gemini3 training dataset

../../azure_tools/azure_run.py --experiment=DR_Qwen_2 --name=run_training --compute=H100x1 \
  -- ./train.py --model_id=Qwen/Qwen3-VL-4B-Instruct --run_id=FineTuned/DR_Qwen \
  --epochs 5 --training_group=training

