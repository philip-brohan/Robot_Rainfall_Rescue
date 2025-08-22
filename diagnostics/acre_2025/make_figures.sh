#!/bin/bash

# Make the diagnostic figures used in my 2025 ACRE talk

# Requires all the necessary model training and test extraction to have been already done
#  and downloaded from Azure.

# SmolVLM cases

# How well does raw SmolVLM do on the test cases?
../../models/gemma/stats_by_label.py --model_id=HuggingFaceTB/SmolVLM-Instruct > smolvlm_instruct_stats.txt

# Plot a good raw smolVLM example
../../models/gemma/plot_image+extracted.py --model_id=HuggingFaceTB/SmolVLM-Instruct --label=TYRain_1941-1950_21_pt1-page-230
mv extracted.webp smolvlm_instruct_good_example.png

# Plot a poor raw smolVLM example
../../models/gemma/plot_image+extracted.py --model_id=HuggingFaceTB/SmolVLM-Instruct --label=TYRain_1951-1960_30_pt1-page-030
mv extracted.webp smolvlm_instruct_bad_example.png

# Plot the stats of the raw SmolVLM model
../../models/gemma/plot_stats.py --model_id=HuggingFaceTB/SmolVLM-Instruct
mv stats.webp smolvlm_instruct_stats.png

# How well does fine-tuned SmolVLM do on the test cases?
../../models/gemma/stats_by_label.py --model_id=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_1/real/nm_1000/rs_42/merged_epoch_12 > smolvlm_trained_stats.txt

# Plot a good raw smolVLM example
../../models/gemma/plot_image+extracted.py --model_id=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_1/real/nm_1000/rs_42/merged_epoch_12 --label=TYRain_1941-1950_21_pt1-page-230
mv extracted.webp smolvlm_trained_good_example.png

# Plot a poor raw smolVLM example
../../models/gemma/plot_image+extracted.py --model_id=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_1/real/nm_1000/rs_42/merged_epoch_12 --label=TYRain_1951-1960_30_pt1-page-030
mv extracted.webp smolvlm_trained_bad_example.png

# Plot the stats of the trained SmolVLM model
../../models/gemma/plot_stats.py --model_id=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_1/real/nm_1000/rs_42/merged_epoch_12
mv stats.webp smolvlm_trained_stats.png

../../models/gemma/plot_stats_comparison.py --model_id_1=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_1/real/nm_1000/rs_42/merged_epoch_12 --model_id_2=HuggingFaceTB/SmolVLM-Instruct
mv stats_comparison.webp smolvlm_stats_comparison.png