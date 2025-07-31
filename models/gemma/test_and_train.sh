#!/bin/bash

# Run the set of jobs to test the gemma models, fine-tune them, and test them again

# Don't just run this script - the earlier jobs must complete before the later ones start
#  I should turn this into some sort of pipeline

echo "Don't just run this script - you need to time it manually"
echo "I just use it to hold the commands and run them with cut and paste"
exit 1

# Do everything twice, once with 3b and once with 12b

icount=2 # Increment this before running any azure jobs, so that the names of the jobs are unique

# Extract fake test cases with the original model
../../azure_tools/azure_run.py --experiment=test_and_train_1 --name=extract_4b_$icount --compute=H100x1 -- ./extract_multi.py --model_id=google/gemma-3-4b-it --fake --image_count=100 --random_seed=42
../../azure_tools/azure_run.py --experiment=test_and_train_1 --name=extract_12bx_$icount --compute=H100x1 -- ./extract_multi.py --model_id=google/gemma-3-12b-it --fake --image_count=100 --random_seed=42

# Extract real test cases with the original model
../../azure_tools/azure_run.py --experiment=test_and_train_1 --name=extract_4b_r_$icount --compute=H100x1 -- ./extract_multi.py --model_id=google/gemma-3-4b-it --image_count=100 --random_seed=42
../../azure_tools/azure_run.py --experiment=test_and_train_1 --name=extract_12b_r_$icount --compute=H100x1 -- ./extract_multi.py --model_id=google/gemma-3-12b-it --image_count=100 --random_seed=42

# Wait for the extractions to finish (6 hours?)

# Copy the original test results to the local system
../../azure_tools/azure_download.py --remote=Robot_Rainfall_Rescue/extracted/google/gemma-3-4b-it/ --local=$PDIR/extracted/google/gemma-3-4b-it/
../../azure_tools/azure_download.py --remote=Robot_Rainfall_Rescue/extracted/google/gemma-3-12b-it/ --local=$PDIR/extracted/google/gemma-3-12b-it/

# Train the model on the fake data
../../azure_tools/azure_run.py --experiment=test_and_train_1 --name=train_4b_$icount --compute=H100x1 -- ./train.py --model_id=google/gemma-3-4b-it --fake --nmax=1000 --random_seed=42 --epochs=3 --run_id=g4b_fake_1000_3
../../azure_tools/azure_run.py --experiment=test_and_train_1 --name=train_12b_$icount --compute=H100x1 -- ./train.py --model_id=google/gemma-3-12b-it --fake --nmax=1000 --random_seed=42 --epochs=3 --run_id=g12b_fake_1000_3

# Wait for the training to finish to finish (6 hours?)

# Extract fake test cases with the fine-tuned model
../../azure_tools/azure_run.py --experiment=test_and_train_1 --name=extract_fte3_4b_$icount --compute=H100x1 -- ./extract_multi.py --model_id=g4b_fake_1000_3 --fake --image_count=100 --random_seed=42 
../../azure_tools/azure_run.py --experiment=test_and_train_1 --name=extract_fte3_12b_$icount --compute=H100x1 -- ./extract_multi.py --model_id=g12b_fake_1000_3 --fake --image_count=100 --random_seed=42 

# Extract real test cases with the fine-tuned model
../../azure_tools/azure_run.py --experiment=test_and_train_1 --name=extract_fte3_4b_r_$icount --compute=H100x1 -- ./extract_multi.py --model_id=g4b_fake_1000_3 --image_count=100 --random_seed=42 
../../azure_tools/azure_run.py --experiment=test_and_train_1 --name=extract_fte3_12b_r_$icount --compute=H100x1 -- ./extract_multi.py --model_id=g12b_fake_1000_3 --image_count=100 --random_seed=42 

# Wait for the extractions to finish  (6 hours?)

# Copy the fine-tuned test results to the local system
../../azure_tools/azure_download.py --remote=Robot_Rainfall_Rescue/extracted/g4b_fake_1000_3 --local=$PDIR/extracted/g4b_fake_1000_3
../../azure_tools/azure_download.py --remote=Robot_Rainfall_Rescue/extracted/g12b_fake_1000_3 --local=$PDIR/extracted/g12b_fake_1000_3
