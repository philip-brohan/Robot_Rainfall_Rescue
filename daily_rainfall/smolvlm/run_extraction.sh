#!/bin/bash

# Train SmolVLM against the Gemini3 training dataset

../../azure_tools/azure_run.py --experiment=DR_SmolVLM --name=run_extraction --compute=A100x1 -- ./extract_group.py --model_id FineTuned/DR_SmolVLM --group validation --batch_size 5 --epoch 10

