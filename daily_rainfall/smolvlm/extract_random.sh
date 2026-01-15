#!/bin/bash

# Run a single test extraction

# HuggingFaceTB/SmolVLM2-2.2B-Instruct
# DRain_1901-1910_RainNos_Kent_Box2_H-W_B112/DRain_1901-1910_RainNos_Kent_Box2_H-W_B112-10 - works well
# DRain_1911-1920_RainNos_Suffolk_B028/DRain_1911-1920_RainNos_Suffolk_B028-3 - works badly

../../azure_tools/azure_run.py --experiment=DR_SmolVLM --name=run_extraction --compute=A100x1 -- ./extract.py --model_id=FineTuned/DR_SmolVLM/merged_epoch_6 --image=DRain_1901-1910_RainNos_Kent_Box2_H-W_B112/DRain_1901-1910_RainNos_Kent_Box2_H-W_B112-10

