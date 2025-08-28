#!/bin/bash

# Make the diagnostic figures used in my 2025 ACRE talk

# Requires all the necessary model training and test extraction to have been already done
#  and downloaded from Azure.

# SmolVLM cases

# How well does raw SmolVLM do on the test cases?
../../models/gemma/stats_by_label.py --model_id=HuggingFaceTB/SmolVLM-Instruct > smolvlm_instruct_stats.txt

# Plot a good raw smolVLM example
../../models/gemma/plot_image+extracted.py --model_id=HuggingFaceTB/SmolVLM-Instruct --label=TYRain_1941-1950_21_pt1-page-230
mv extracted.webp smolvlm_instruct_good_example.webp

# Plot a poor raw smolVLM example
../../models/gemma/plot_image+extracted.py --model_id=HuggingFaceTB/SmolVLM-Instruct --label=TYRain_1951-1960_30_pt1-page-030
mv extracted.webp smolvlm_instruct_bad_example.webp

# Plot the stats of the raw SmolVLM model
../../models/gemma/plot_stats.py --model_id=HuggingFaceTB/SmolVLM-Instruct
mv stats.webp smolvlm_instruct_stats.webp

# How well does fine-tuned SmolVLM do on the test cases?
../../models/gemma/stats_by_label.py --model_id=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_1/real/nm_1000/rs_42/merged_epoch_12 > smolvlm_trained_stats.txt

# show the effect of training SmolVLM on an easy example
../../models/gemma/plot_image+comparison.py --model_id_1=HuggingFaceTB/SmolVLM-Instruct --model_id_2=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_1/real/nm_1000/rs_42/merged_epoch_12 --label=TYRain_1941-1950_21_pt1-page-230
mv compare.webp smolvlm_trained_good_example.webp

# show the effect of training SmolVLM on an difficult example
../../models/gemma/plot_image+comparison.py --model_id_1=HuggingFaceTB/SmolVLM-Instruct --model_id_2=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_1/real/nm_1000/rs_42/merged_epoch_12 --label=TYRain_1951-1960_30_pt1-page-030
mv compare.webp smolvlm_trained_bad_example.webp

# Plot the stats of the trained SmolVLM model
../../models/gemma/plot_stats.py --model_id=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_1/real/nm_1000/rs_42/merged_epoch_12
mv stats.webp smolvlm_trained_stats.webp

../../models/gemma/plot_stats_comparison.py --model_id_1=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_1/real/nm_1000/rs_42/merged_epoch_12 --model_id_2=HuggingFaceTB/SmolVLM-Instruct
mv stats_comparison.webp smolvlm_stats_comparison.webp

# Show 3-model trained agreement on an easy example
../../models/gemma/plot_agreement.py --model_ids=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_1/real/nm_1000/rs_42/merged_epoch_12,FineTuned/google/gemma-3-4b-it/hpb_1/real/nm_1000/rs_42/merged_epoch_12,FineTuned/ibm-granite/granite-vision-3.3-2b/hpb_1/real/nm_1000/rs_42/merged_epoch_12 --label=TYRain_1941-1950_21_pt1-page-230
mv agree.webp smolvlm_gemma_granite_agreement_easy.webp

# Show 3-model trained agreement on a difficult example
../../models/gemma/plot_agreement.py --model_ids=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_1/real/nm_1000/rs_42/merged_epoch_12,FineTuned/google/gemma-3-4b-it/hpb_1/real/nm_1000/rs_42/merged_epoch_12,FineTuned/ibm-granite/granite-vision-3.3-2b/hpb_1/real/nm_1000/rs_42/merged_epoch_12 --label=TYRain_1951-1960_30_pt1-page-030
mv agree.webp smolvlm_gemma_granite_agreement_hard.webp

# show 3-model trained agreement stats
../../models/gemma/plot_stats_agreement.py --model_ids=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_1/real/nm_1000/rs_42/merged_epoch_12,FineTuned/google/gemma-3-4b-it/hpb_1/real/nm_1000/rs_42/merged_epoch_12,FineTuned/ibm-granite/granite-vision-3.3-2b/hpb_1/real/nm_1000/rs_42/merged_epoch_12
mv stats_agrement.webp smolvlm_gemma_granite_agreement_stats.webp

# show 3-model untrained agreement stats
../../models/gemma/plot_stats_agreement.py --model_ids=HuggingFaceTB/SmolVLM-Instruct,google/gemma-3-4b-it,ibm-granite/granite-vision-3.3-2b
mv stats_agrement.webp smolvlm_gemma_granite_untrained_agreement_stats.webp