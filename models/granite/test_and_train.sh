#!/bin/bash

# Run the set of jobs to test the granite model, fine-tune it, and test it again

# Don't just run this script - the earlier jobs must complete before the later ones start
#  I should turn this into some sort of pipeline

# Each job must have a globally unique name. Increment this number to avoid conflicts
#  when running this script multiple times - it's only used in job names (no effect on outputs).
icount=3

# Hyperparameter batch number
#  This is used to distinguish between different hyperparameter settings in training
#  Hyperparameters are set in the train.py script
hpbatch=1

# We're doing 12 epochs of training. Which epoch are we using for testing?
epoch=12

# Extract fake test cases with the original model
../../azure_tools/azure_run.py --experiment=granite_1 --name=extract_gr_f_$icount --compute=H100x1 -- ./extract_multi.py --model_id=ibm-granite/granite-vision-3.3-2b --fake --image_count=100 --random_seed=42
# Extract real test cases with the original model
../../azure_tools/azure_run.py --experiment=granite_1 --name=extract_gr_r_$icount --compute=H100x1 -- ./extract_multi.py --model_id=ibm-granite/granite-vision-3.3-2b --image_count=100 --random_seed=42

# Wait for the extractions to finish (1 hours)

# Copy the fake test results to the local system
../../azure_tools/azure_download.py --remote=Robot_Rainfall_Rescue/extracted/ibm-granite/granite-vision-3.3-2b --local=$PDIR/extracted/ibm-granite/granite-vision-3.3-2b

# Train the model on the fake data
../../azure_tools/azure_run.py --experiment=granite_1 --name=train_gr_f_$icount --compute=H100x1 -- ./train.py --model_id=ibm-granite/granite-vision-3.3-2b --fake --nmax=1000 --random_seed=42 --epochs=12 --run_id=FineTuned/ibm-granite/granite-vision-3.3-2b/hpb_$hpbatch/fake/nm_1000/rs_42

# Train the model on real data
../../azure_tools/azure_run.py --experiment=granite_1 --name=train_gr_r_$icount --compute=H100x1 -- ./train.py --model_id=ibm-granite/granite-vision-3.3-2b --nmax=1000 --random_seed=42 --epochs=12 --run_id=FineTuned/ibm-granite/granite-vision-3.3-2b/hpb_$hpbatch/real/nm_1000/rs_42

# Wait for the training to finish to finish (6 hours?)

# Extract fake test cases with the model fine-tuned on fake data
../../azure_tools/azure_run.py --experiment=granite_1 --name=ex_gr_f_$hpbatch\_f_$epoch\_$icount --compute=H100x1 -- ./extract_multi.py --model_id=FineTuned/ibm-granite/granite-vision-3.3-2b/hpb_$hpbatch/fake/nm_1000/rs_42/merged_epoch_$epoch --fake --image_count=100 --random_seed=42
# Extract real test cases with the model fine-tuned on fake data
../../azure_tools/azure_run.py --experiment=granite_1 --name=ex_gr_f_$hpbatch\_r_$epoch\_$icount --compute=H100x1 -- ./extract_multi.py --model_id=FineTuned/ibm-granite/granite-vision-3.3-2b/hpb_$hpbatch/fake/nm_1000/rs_42/merged_epoch_$epoch --image_count=100 --random_seed=42
# Extract fake test cases with the model fine-tuned on real data
../../azure_tools/azure_run.py --experiment=granite_1 --name=ex_gr_r_$hpbatch\_f_$epoch\_$icount --compute=H100x1 -- ./extract_multi.py --model_id=FineTuned/ibm-granite/granite-vision-3.3-2b/hpb_$hpbatch/real/nm_1000/rs_42/merged_epoch_$epoch --fake --image_count=100 --random_seed=42
# Extract real test cases with the model fine-tuned on real data
../../azure_tools/azure_run.py --experiment=granite_1 --name=ex_gr_r_$hpbatch\_r_$epoch\_$icount --compute=H100x1 -- ./extract_multi.py --model_id=FineTuned/ibm-granite/granite-vision-3.3-2b/hpb_$hpbatch/real/nm_1000/rs_42/merged_epoch_$epoch --image_count=100 --random_seed=42

# Wait for the extractions to finish (1 hours)

# Copy the fake test results to the local system
../../azure_tools/azure_download.py --remote=Robot_Rainfall_Rescue/extracted/FineTuned/ibm-granite/granite-vision-3.3-2b/ --local=$PDIR/extracted/FineTuned/ibm-granite/granite-vision-3.3-2b/
