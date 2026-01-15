#!/bin/bash

# Train SmolVLM against the Gemini3 training dataset

# HuggingFaceTB/SmolVLM2-2.2B-Instruct

../../azure_tools/azure_run.py --experiment=DR_SmolVLM --name=run_training --compute=A100x1 -- ./train.py --model_id=HuggingFaceTB/SmolVLM-Instruct --run_id=FineTuned/DR_SmolVLM --epochs 10

