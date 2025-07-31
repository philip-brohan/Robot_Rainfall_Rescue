#!/bin/bash

# Run the set of jobs to test the SmolVLM model, fine-tune it, and test it again

# Don't just run this script - the earlier jobs must complete before the later ones start
#  I should turn this into some sort of pipeline

icount=1

# Extract fake test cases with the original model
../../azure_tools/azure_run.py --experiment=test_and_train_1 --name=extract_sm_$icount --compute=H100x1 -- ./extract_multi.py --model_id=HuggingFaceTB/SmolVLM-Instruct --fake --image_count=100 --random_seed=42

# Wait for the extractions to finish (6 hours?)

# Copy the fake test results to the local system
../../azure_tools/azure_download.py --remote=Robot_Rainfall_Rescue/extracted/HuggingFaceTB/SmolVLM-Instruct --local=$PDIR/extracted/HuggingFaceTB/SmolVLM-Instruct

# Train the model on the fake data
#../../azure_tools/azure_run.py --experiment=test_and_train_1 --name=train_sm_$icount --compute=H100x1 -- ./train.py --model_id=HuggingFaceTB/SmolVLM-Instruct --fake --nmax=1000 --random_seed=42 --epochs=3 --run_id=sm_fake_1000_3

# Wait for the training to finish to finish (Doesn't yet work at all)

# Extract fake test cases with the fine-tuned model
#../../azure_tools/azure_run.py --experiment=test_and_train_1 --name=extract_fte3_sm_$icount --compute=H100x1 -- ./extract_multi.py --model_id=sm_fake_1000_3 --fake --image_count=100 --random_seed=42

# Wait for the extractions to finish (6 hours?)

# Copy the fake test results to the local system
../../azure_tools/azure_download.py --remote=Robot_Rainfall_Rescue/extracted/sm_fake_1000_3 --local=$PDIR/extracted/sm_fake_1000_3
