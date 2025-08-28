#!/bin/bash

# Run the set of jobs to test the gemma models, fine-tune them, and test them again

# Don't just run this script - the earlier jobs must complete before the later ones start
#  I should turn this into some sort of pipeline

echo "Don't just run this script - you need to time it manually"
echo "I just use it to hold the commands and run them with cut and paste"
exit 1


# Each job must have a globally unique name. Increment this number to avoid conflicts
#  when running this script multiple times - it's only used in job names (no effect on outputs).
icount=10 

# which size model to use
size="4b" # 4b or 12b - bigger ones don't it on the GPU

# Hyperparameter batch number
#  This is used to distinguish between different hyperparameter settings in training
#  Hyperparameters are set in the train.py script
hpbatch=1

# We're doing 12 epochs of training. Which epoch are we using for testing?
epoch=12

# Extract fake test cases with the original model
../../azure_tools/azure_run.py --experiment=gemma_$size\_1 --name=extract_gm_$size\_f_$icount --compute=H100x1 -- ./extract_multi.py --model_id=google/gemma-3-$size-it --fake --image_count=100 --random_seed=42

# Extract real test cases with the original model
../../azure_tools/azure_run.py --experiment=gemma_$size\_1 --name=extract_gm_$size\_r_$icount --compute=H100x1 -- ./extract_multi.py --model_id=google/gemma-3-$size-it --image_count=100 --random_seed=42

# Wait for the extractions to finish (6 hours?)

# Copy the original test results to the local system
../../azure_tools/azure_download.py --remote=Robot_Rainfall_Rescue/extracted/google/gemma-3-$size-it/ --local=$PDIR/extracted/google/gemma-3-$size-it/

# Train the model on the fake data
../../azure_tools/azure_run.py --experiment=gemma_$size\_1 --name=train_gm_$size\_f_$icount --compute=H100x1 -- ./train.py --model_id=google/gemma-3-$size-it --fake --nmax=1000 --random_seed=42 --epochs=12 --run_id=FineTuned/google/gemma-3-$size-it/hpb_$hpbatch/fake/nm_1000/rs_42

# Train the model on the real data
../../azure_tools/azure_run.py --experiment=gemma_$size\_1 --name=train_gm_$size\_r_$icount --compute=H100x1 -- ./train.py --model_id=google/gemma-3-$size-it --nmax=1000 --random_seed=42 --epochs=12 --run_id=FineTuned/google/gemma-3-$size-it/hpb_$hpbatch/real/nm_1000/rs_42

# Wait for the training to finish to finish (6 hours?)

# Extract fake test cases with the model fine-tuned on fake data
../../azure_tools/azure_run.py --experiment=gemma_$size\_1 --name=ex_gm_$size\_f_$hpbatch\_f_$epoch\_$icount --compute=H100x1 -- ./extract_multi.py --model_id=FineTuned/google/gemma-3-$size\-it/hpb_$hpbatch/fake/nm_1000/rs_42/merged_epoch_$epoch --fake --image_count=100 --random_seed=42 
# Extract real test cases with the model fine-tuned on fake data
../../azure_tools/azure_run.py --experiment=gemma_$size\_1 --name=ex_gm_$size\_f_$hpbatch\_r_$epoch\_$icount --compute=H100x1 -- ./extract_multi.py --model_id=FineTuned/google/gemma-3-$size-it/hpb_$hpbatch/fake/nm_1000/rs_42/merged_epoch_$epoch --image_count=100 --random_seed=42 
# Extract fake test cases with the model fine-tuned on real data
../../azure_tools/azure_run.py --experiment=gemma_$size\_1 --name=ex_gm_$size\_r_$hpbatch\_f_$epoch\_$icount --compute=H100x1 -- ./extract_multi.py --model_id=FineTuned/google/gemma-3-$size-it/hpb_$hpbatch/real/nm_1000/rs_42/merged_epoch_$epoch --fake --image_count=100 --random_seed=42 
# Extract real test cases with the model fine-tuned on real data
../../azure_tools/azure_run.py --experiment=gemma_$size\_1 --name=ex_gm_$size\_r_$hpbatch\_r_$epoch\_$icount --compute=H100x1 -- ./extract_multi.py --model_id=FineTuned/google/gemma-3-$size-it/hpb_$hpbatch/real/nm_1000/rs_42/merged_epoch_$epoch --image_count=100 --random_seed=42

# Wait for the extractions to finish  (30 minutes?)

# Copy the fine-tuned test results to the local system
../../azure_tools/azure_download.py --remote=Robot_Rainfall_Rescue/extracted/FineTuned/google/gemma-3-$size-it --local=$PDIR/extracted/FineTuned/google/gemma-3-$size-it
