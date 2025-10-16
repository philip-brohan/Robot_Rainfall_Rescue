#!/bin/bash

# Make the diagnostic figures used in my 2025 ACRE talk

# Requires all the necessary model training and test extraction to have been already done
#  and downloaded from Azure.


# How well do the raw models do on the test cases?
../../models/gemma/stats_by_label.py --model_id=HuggingFaceTB/SmolVLM-Instruct > smolvlm_stats.txt
../../models/gemma/stats_by_label.py --model_id=ibm-granite/granite-vision-3.3-2b > granite_stats.txt
../../models/gemma/stats_by_label.py --model_id=google/gemma-3-4b-it > gemma_stats.txt

# Plot a good raw smolVLM example
../../models/gemma/plot_image+extracted.py --model_id=HuggingFaceTB/SmolVLM-Instruct --label=TYRain_1910-1919_19_pt1-page-130
mv extracted.webp smolvlm_good_example.webp

# Plot a poor raw smolVLM example
../../models/gemma/plot_image+extracted.py --model_id=HuggingFaceTB/SmolVLM-Instruct --label=TYRain_1910-1919_19_pt1-page-210
mv extracted.webp smolvlm_bad_example.webp

# Plot the same examples for granite
../../models/gemma/plot_image+extracted.py --model_id=ibm-granite/granite-vision-3.3-2b --label=TYRain_1910-1919_19_pt1-page-130
mv extracted.webp granite_good_example.webp
../../models/gemma/plot_image+extracted.py --model_id=ibm-granite/granite-vision-3.3-2b --label=TYRain_1910-1919_19_pt1-page-210
mv extracted.webp granite_bad_example.webp

# Plot the same examples for gemma
../../models/gemma/plot_image+extracted.py --model_id=google/gemma-3-4b-it --label=TYRain_1910-1919_19_pt1-page-130
mv extracted.webp gemma_good_example.webp
../../models/gemma/plot_image+extracted.py --model_id=google/gemma-3-4b-it --label=TYRain_1910-1919_19_pt1-page-210
mv extracted.webp gemma_bad_example.webp

# Plot the stats of the raw models
../../models/gemma/plot_stats.py --model_id=HuggingFaceTB/SmolVLM-Instruct
mv stats.webp smolvlm_stats.webp
../../models/gemma/plot_stats.py --model_id=ibm-granite/granite-vision-3.3-2b
mv stats.webp granite_stats.webp
../../models/gemma/plot_stats.py --model_id=google/gemma-3-4b-it   
mv stats.webp gemma_stats.webp

# show 3-model untrained agreement stats
../../models/gemma/plot_stats_agreement.py --model_ids=HuggingFaceTB/SmolVLM-Instruct,google/gemma-3-4b-it,ibm-granite/granite-vision-3.3-2b
mv stats_agreement.webp smolvlm_gemma_granite_untrained_agreement_stats.webp

# How well does fine-tuned SmolVLM do on the test cases?
../../models/gemma/stats_by_label.py --model_id=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_2/real/nm_1000/rs_42/merged_epoch_12 > smolvlm_trained_stats.txt

# Show the effect of training SmolVLM on an easy example
../../models/gemma/plot_image+comparison.py --model_id_1=HuggingFaceTB/SmolVLM-Instruct --model_id_2=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_2/real/nm_1000/rs_42/merged_epoch_12 --label=TYRain_1910-1919_19_pt1-page-130
mv compare.webp smolvlm_trained_good_example.webp

# Show the effect of training SmolVLM on an difficult example
../../models/gemma/plot_image+comparison.py --model_id_1=HuggingFaceTB/SmolVLM-Instruct --model_id_2=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_2/real/nm_1000/rs_42/merged_epoch_12 --label=TYRain_1910-1919_19_pt1-page-210
mv compare.webp smolvlm_trained_bad_example.webp

# Plot the stats of the trained SmolVLM model
../../models/gemma/plot_stats.py --model_id=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_2/real/nm_1000/rs_42/merged_epoch_12
mv stats.webp smolvlm_trained_stats.webp
../../models/gemma/plot_stats_comparison.py --model_id_2=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_2/real/nm_1000/rs_42/merged_epoch_12 --model_id_1=HuggingFaceTB/SmolVLM-Instruct
mv stats_comparison.webp smolvlm_stats_comparison.webp

# Plot the stats of the SmolVLM model trained on fake data
../../models/gemma/plot_stats.py --model_id=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_2/fake/nm_1000/rs_42/merged_epoch_12
mv stats.webp smolvlm_trained_fake_stats.webp
../../models/gemma/plot_stats_comparison.py --model_id_2=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_2/fake/nm_1000/rs_42/merged_epoch_12 --model_id_1=HuggingFaceTB/SmolVLM-Instruct
mv stats_comparison.webp smolvlm_fake_stats_comparison.webp

# Show the effect of training Granite on an easy example
../../models/gemma/plot_image+comparison.py --model_id_1=ibm-granite/granite-vision-3.3-2b --model_id_2=FineTuned/ibm-granite/granite-vision-3.3-2b/hpb_2/real/nm_1000/rs_42/merged_epoch_12 --label=TYRain_1910-1919_19_pt1-page-130
mv compare.webp granite_trained_good_example.webp

# Show the effect of training Granite on a difficult example
../../models/gemma/plot_image+comparison.py --model_id_1=ibm-granite/granite-vision-3.3-2b --model_id_2=FineTuned/ibm-granite/granite-vision-3.3-2b/hpb_2/real/nm_1000/rs_42/merged_epoch_12 --label=TYRain_1910-1919_19_pt1-page-210
mv compare.webp granite_trained_bad_example.webp

# Plot the stats of the trained Granite model
../../models/gemma/plot_stats.py --model_id=FineTuned/ibm-granite/granite-vision-3.3-2b/hpb_2/real/nm_1000/rs_42/merged_epoch_12
mv stats.webp granite_trained_stats.webp
../../models/gemma/plot_stats_comparison.py --model_id_2=FineTuned/ibm-granite/granite-vision-3.3-2b/hpb_2/real/nm_1000/rs_42/merged_epoch_12 --model_id_1=ibm-granite/granite-vision-3.3-2b
mv stats_comparison.webp granite_stats_comparison.webp

# Plot the stats of the Granite model trained on fake data
../../models/gemma/plot_stats.py --model_id=FineTuned/ibm-granite/granite-vision-3.3-2b/hpb_2/fake/nm_1000/rs_42/merged_epoch_12
mv stats.webp granite_trained_fake_stats.webp
../../models/gemma/plot_stats_comparison.py --model_id_2=FineTuned/ibm-granite/granite-vision-3.3-2b/hpb_2/fake/nm_1000/rs_42/merged_epoch_12 --model_id_1=ibm-granite/granite-vision-3.3-2b
mv stats_comparison.webp granite_fake_stats_comparison.webp

# Show the effect of training Gemma on an easy example
../../models/gemma/plot_image+comparison.py --model_id_1=google/gemma-3-4b-it --model_id_2=FineTuned/google/gemma-3-4b-it/hpb_2/real/nm_1000/rs_42/merged_epoch_12 --label=TYRain_1910-1919_19_pt1-page-130
mv compare.webp gemma_trained_good_example.webp

# Show the effect of training Gemma on a difficult example
../../models/gemma/plot_image+comparison.py --model_id_1=google/gemma-3-4b-it --model_id_2=FineTuned/google/gemma-3-4b-it/hpb_2/real/nm_1000/rs_42/merged_epoch_12 --label=TYRain_1910-1919_19_pt1-page-210
mv compare.webp gemma_trained_bad_example.webp

# Plot the stats of the trained Gemma model
../../models/gemma/plot_stats.py --model_id=FineTuned/google/gemma-3-4b-it/hpb_2/real/nm_1000/rs_42/merged_epoch_12
mv stats.webp gemma_trained_stats.webp
../../models/gemma/plot_stats_comparison.py --model_id_2=FineTuned/google/gemma-3-4b-it/hpb_2/real/nm_1000/rs_42/merged_epoch_12 --model_id_1=google/gemma-3-4b-it
mv stats_comparison.webp gemma_stats_comparison.webp

# Plot the stats of the Gemma model trained on fake data
../../models/gemma/plot_stats.py --model_id=FineTuned/google/gemma-3-4b-it/hpb_2/fake/nm_1000/rs_42/merged_epoch_12
mv stats.webp gemma_trained_fake_stats.webp
../../models/gemma/plot_stats_comparison.py --model_id_2=FineTuned/google/gemma-3-4b-it/hpb_2/fake/nm_1000/rs_42/merged_epoch_12 --model_id_1=google/gemma-3-4b-it
mv stats_comparison.webp gemma_fake_stats_comparison.webp

# Show 3-model untrained agreement on an easy example
../../models/gemma/plot_agreement.py --model_ids=HuggingFaceTB/SmolVLM-Instruct,google/gemma-3-4b-it,ibm-granite/granite-vision-3.3-2b --label=TYRain_1910-1919_19_pt1-page-130
mv agree.webp smolvlm_gemma_granite_untrained_agreement_easy.webp

# Show 3-model untrained agreement on an difficult example
../../models/gemma/plot_agreement.py --model_ids=HuggingFaceTB/SmolVLM-Instruct,google/gemma-3-4b-it,ibm-granite/granite-vision-3.3-2b --label=TYRain_1910-1919_19_pt1-page-210
mv agree.webp smolvlm_gemma_granite_untrained_agreement_hard.webp

# Show 3-model trained agreement on an easy example
../../models/gemma/plot_agreement.py --model_ids=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_2/real/nm_1000/rs_42/merged_epoch_12,FineTuned/google/gemma-3-4b-it/hpb_2/real/nm_1000/rs_42/merged_epoch_12,FineTuned/ibm-granite/granite-vision-3.3-2b/hpb_2/real/nm_1000/rs_42/merged_epoch_12 --label=TYRain_1910-1919_19_pt1-page-130
mv agree.webp smolvlm_gemma_granite_agreement_easy.webp

# Show 3-model trained agreement on a difficult example
../../models/gemma/plot_agreement.py --model_ids=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_2/real/nm_1000/rs_42/merged_epoch_12,FineTuned/google/gemma-3-4b-it/hpb_2/real/nm_1000/rs_42/merged_epoch_12,FineTuned/ibm-granite/granite-vision-3.3-2b/hpb_2/real/nm_1000/rs_42/merged_epoch_12 --label=TYRain_1910-1919_19_pt1-page-210
mv agree.webp smolvlm_gemma_granite_agreement_hard.webp

# show 3-model trained agreement stats
../../models/gemma/plot_stats_agreement.py --model_ids=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_2/real/nm_1000/rs_42/merged_epoch_12,FineTuned/google/gemma-3-4b-it/hpb_2/real/nm_1000/rs_42/merged_epoch_12,FineTuned/ibm-granite/granite-vision-3.3-2b/hpb_2/real/nm_1000/rs_42/merged_epoch_12
mv stats_agreement.webp smolvlm_gemma_granite_agreement_stats.webp

# show 3-model agreement stats trained on fake data
../../models/gemma/plot_stats_agreement.py --model_ids=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_2/fake/nm_1000/rs_42/merged_epoch_12,FineTuned/google/gemma-3-4b-it/hpb_2/fake/nm_1000/rs_42/merged_epoch_12,FineTuned/ibm-granite/granite-vision-3.3-2b/hpb_2/fake/nm_1000/rs_42/merged_epoch_12
mv stats_agreement.webp smolvlm_gemma_granite_agreement_fake_stats.webp

# show 3-model agreement stats with smolvlm and gemma trained on fake data
../../models/gemma/plot_stats_agreement.py --model_ids=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_2/fake/nm_1000/rs_42/merged_epoch_12,FineTuned/google/gemma-3-4b-it/hpb_2/fake/nm_1000/rs_42/merged_epoch_12,ibm-granite/granite-vision-3.3-2b --agreement_count=3
mv stats_agreement.webp smolvlm_gemma_granite_agreement_fake_stats.webp
