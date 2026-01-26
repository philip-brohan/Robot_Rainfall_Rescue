#!/bin/bash

# Validate fine-tuned Qwen3 4B against the Gemini3 validation dataset

../../azure_tools/azure_run.py --experiment=DR_Qwen --name=run_validation --compute=A100x1 -- ./validate.py --base_model_id=Qwen/Qwen3-VL-4B-Instruct --model_id=FineTuned/DR_Qwen/merged_epoch_3 --validation_group=validation
