#!/bin/bash

# Make demo images for validation script documentation

../../../models/gemma/plot_image+extracted.py --model=google/gemma-3-4b-it --label=TYRain_1900-1909_14_pt1-page-040

../../../models/gemma/plot_stats.py --model=google/gemma-3-4b-it --purpose=Test

../../../models/gemma/plot_agreement.py --model=HuggingFaceTB/SmolVLM-Instruct,google/gemma-3-4b-it,ibm-granite/granite-vision-3.3-2b --label=TYRain_1900-1909_14_pt1-page-040

../../../models/gemma/plot_stats_agreement.py --model=HuggingFaceTB/SmolVLM-Instruct,google/gemma-3-4b-it,ibm-granite/granite-vision-3.3-2b --purpose=Test --agreement_count=2

../../../models/gemma/plot_image+comparison.py --model_id_1=HuggingFaceTB/SmolVLM-Instruct --model_id_2=FineTuned/HuggingFaceTB/SmolVLM-Instruct/hpb_2/real/nm_1000/rs_42/merged_epoch_12  --label=TYRain_1900-1909_14_pt1-page-040