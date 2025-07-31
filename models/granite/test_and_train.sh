#!/bin/bash

# Run the set of jobs to test the granite model, fine-tune it, and test it again

# Don't just run this script - the earlier jobs must complete before the later ones start
#  I should turn this into some sort of pipeline

icount=2

# Extract fake test cases with the original model
../../azure_tools/azure_run.py --experiment=test_and_train_1 --name=extract_gr_$icount --compute=H100x1 -- ./extract_multi.py --model_id=ibm-granite/granite-vision-3.3-2b --fake --image_count=100 --random_seed=42
# Extract real test cases with the original model
../../azure_tools/azure_run.py --experiment=test_and_train_1 --name=extract_gr_r_$icount --compute=H100x1 -- ./extract_multi.py --model_id=ibm-granite/granite-vision-3.3-2b --image_count=100 --random_seed=42

# Wait for the extractions to finish (1 hours)

# Copy the fake test results to the local system
../../azure_tools/azure_download.py --remote=Robot_Rainfall_Rescue/extracted/ibm-granite/granite-vision-3.3-2b --local=$PDIR/extracted/ibm-granite/granite-vision-3.3-2b

# Train the model on the fake data
../../azure_tools/azure_run.py --experiment=test_and_train_1 --name=train_gr_$icount --compute=H100x1 -- ./train.py --model_id=ibm-granite/granite-vision-3.3-2b --fake --nmax=1000 --random_seed=42 --epochs=3 --run_id=gr_fake_1000_3

# Wait for the training to finish to finish (6 hours?)

# Extract fake test cases with the fine-tuned model
../../azure_tools/azure_run.py --experiment=test_and_train_1 --name=extract_fte3_gr_$icount --compute=H100x1 -- ./extract_multi.py --model_id=gr_fake_1000_3 --fake --image_count=100 --random_seed=42
# Extract real test cases with the fine-tuned model
../../azure_tools/azure_run.py --experiment=test_and_train_1 --name=extract_fte3_gr_r_$icount --compute=H100x1 -- ./extract_multi.py --model_id=gr_fake_1000_3 --image_count=100 --random_seed=42

# Wait for the extractions to finish (1 hours)

# Copy the fake test results to the local system
../../azure_tools/azure_download.py --remote=Robot_Rainfall_Rescue/extracted/gr_fake_1000_3 --local=$PDIR/extracted/gr_fake_1000_3
