#!/bin/bash

# Run the set of jobs to test the SmolVLM model, fine-tune it, and test it again

# Don't just run this script - the earlier jobs must complete before the later ones start
#  I should turn this into some sort of pipeline

# Each job must have a globally unique name. Increment this number to avoid conflicts
#  when running this script multiple times - it's only used in job names (no effect on outputs).
icount=10

# Hyperparameter batch number
#  This is used to distinguish between different hyperparameter settings in training
#  Hyperparameters are set in the train.py script
hpbatch=1

# We're doing 12 epochs of training. Which epoch are we using for testing?
epoch=12

# Extract fake test cases with the original model
../../azure_tools/azure_run.py --experiment=smolvlm_1 --name=extract_sm_$icount --compute=H100x1 -- ./extract_multi.py --model_id=HuggingFaceTB/SmolVLM-Instruct --fake --image_count=100 --random_seed=42

# Extract real test cases with the original model
../../azure_tools/azure_run.py --experiment=smolvlm_1 --name=extract_sm_r_$icount --compute=H100x1 -- ./extract_multi.py --model_id=HuggingFaceTB/SmolVLM-Instruct --image_count=100 --random_seed=42

# Wait for the extractions to finish (30 minutes)

# Copy the test results to the local system
../../azure_tools/azure_download.py --remote=Robot_Rainfall_Rescue/extracted/HuggingFaceTB/SmolVLM-Instruct --local=$PDIR/extracted/HuggingFaceTB/SmolVLM-Instruct

# fine-tune the model on the fake data
../../azure_tools/azure_run.py --experiment=smolvlm_1 --name=train_sm_f_$icount --compute=H100x1 -- ./train.py --model_id=HuggingFaceTB/SmolVLM-Instruct --fake --nmax=1000 --random_seed=42 --epochs=12 --run_id=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_$hpbatch/fake/nm_1000/rs_42

# Fine-tune the model on the real data
../../azure_tools/azure_run.py --experiment=smolvlm_1 --name=train_sm_r_$icount --compute=H100x1 -- ./train.py --model_id=HuggingFaceTB/SmolVLM-Instruct --nmax=1000 --random_seed=42 --epochs=12 --run_id=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_$hpbatch/real/nm_1000/rs_42

# Wait for the training to finish to finish (5 hours?)

# Extract fake test cases with the model fine-tuned on fake data
../../azure_tools/azure_run.py --experiment=smolvlm_1 --name=ex_sm_f_$hpbatch\_f_$epoch\_$icount --compute=H100x1 -- ./extract_multi.py --model_id=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_$hpbatch/fake/nm_1000/rs_42/merged_epoch_$epoch --fake --image_count=100 --random_seed=42
# Extract real test cases with the model fine-tuned on fake data
../../azure_tools/azure_run.py --experiment=smolvlm_1 --name=ex_sm_f_$hpbatch\_r_$epoch\_$icount --compute=H100x1 -- ./extract_multi.py --model_id=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_$hpbatch/fake/nm_1000/rs_42/merged_epoch_$epoch --image_count=100 --random_seed=42

# Extract fake test cases with the model fine-tuned on real data
../../azure_tools/azure_run.py --experiment=smolvlm_1 --name=ex_sm_r_$hpbatch\_f_$epoch\_$icount --compute=H100x1 -- ./extract_multi.py --model_id=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_$hpbatch/real/nm_1000/rs_42/merged_epoch_$epoch --fake --image_count=100 --random_seed=42
# Extract real test cases with the model fine-tuned on real data
../../azure_tools/azure_run.py --experiment=smolvlm_1 --name=ex_sm_r_$hpbatch\_r_$epoch\_$icount --compute=H100x1 -- ./extract_multi.py --model_id=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_$hpbatch/real/nm_1000/rs_42/merged_epoch_$epoch --image_count=100 --random_seed=42

# Wait for the extractions to finish (30 minutes?)

# Copy the test results to the local system
../../azure_tools/azure_download.py --remote=Robot_Rainfall_Rescue/extracted/FineTuned/HuggingFaceTB/SmolVLM-Instruct --local=$PDIR/extracted/FineTuned/HuggingFaceTB/SmolVLM-Instruct
