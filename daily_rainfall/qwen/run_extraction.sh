#!/bin/bash

# Generate daily precipitation from images with fine-tuned Qwen3 4B

../../azure_tools/azure_run.py --experiment=DR_Qwen --name=run_generation --compute=A100x1 -- ./generate.py --base_model_id=Qwen/Qwen3-VL-4B-Instruct --model_id=FineTuned/DR_Qwen/merged_epoch_3 --generation_group=validation --batch_size=2