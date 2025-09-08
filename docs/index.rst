Robot Rainfall Rescue
=====================

Can we use Artificial Intelligence (AI) to rapidly transcribe vital climate data from paper archives? We've `shown that Google Gemini can do this in some cases <https://brohan.org/AI_daily_precip/>`_, but it struggles with tables including missing data, and it doesn't provide good estimates of uncertainty. So we are going to construct a method using multiple small Vision-Language Models (VLMs) - we will get uncertainty estimates from the variation between models, and we will fine-tune them explicitly to deal with missing data.

The problem
-----------

.. figure:: illustrations/Image_samples.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   Three sample images showing monthly average rainfall observations. Each image contains the data from one station for 10 years.

.. toctree::
   :maxdepth: 1

   Download the Rainfall Rescue data <rainfall_rescue/get_data>
   Diagnostic plots for Rainfall_Rescue data <rainfall_rescue/diagnostics>
   Utility functions <rainfall_rescue/utilities>
   Image utilities <RR_utils/image>

Meet the team
-------------

.. figure:: illustrations/smolvlm_granite_gemma.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Three small VLMs, `SmolVLM <https://huggingface.co/blog/smolvlm>`_, `Granite <https://www.ibm.com/granite>`_, and `Gemma <https://deepmind.google/models/gemma/gemma-3/>`_. Note that these depictions are not official images from the model developers, but `anthropomorphic representations created by our artist <https://chatgpt.com/share/68b5a6ff-fc00-8013-b62f-2c4032700aa6>`_.

.. toctree::
   :maxdepth: 1

   Huggingface interface functions <RR_utils/hf>


A single small VLM: SmolVLM
---------------------------

.. figure:: illustrations/SmolVLM_raw_easy.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   SmolVLM performance on a single image. Text in black is where the model is correct, in red shows where it is wrong (and where it is wrong, the correct value is given underneath in blue). This is a relatively easy example, with no missing data.

.. figure:: illustrations/SmolVLM_raw_hard.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   SmolVLM performance on a single image. Text in black is where the model is correct, in red shows where it is wrong (and where it is wrong, the correct value is given underneath in blue). This is a challenging example, with missing data.

.. figure:: illustrations/SmolVLM_raw_stats.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   SmolVLM percentage success rate for each position in the table.

.. toctree::
   :maxdepth: 1

   Prompts used by the models <models/smolvlm/prompts>
   Run SmolVLM on a single example image <models/smolvlm/extract>
   Run SmolVLM on a set of images <models/smolvlm/extract_multi>
   Model output diagnostic scripts <models/gemma/diagnostics>
   


An ensemble of VLMs
-------------------

.. figure:: illustrations/SmolVLM_Granite_Gemma_raw_example.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   Example results from the three-model ensemble (untrained). Values in blue show where 2 or more models agree on a value, and that value is correct. Values in red show where 2 or more models agree on a value, but that value is wrong. Values in grey show where there is no agreement among the models.

.. figure:: illustrations/SmolVLM_Granite_Gemma_raw_stats.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   Percentage success rate for each position in the table, for the three model ensemble. Value in blue is the percentage of values where 2 or more models agree, and that value is correct. Value in red is the percentage of values where 2 or more models agree, but that value is wrong. 

Training the models
-------------------

.. figure:: illustrations/Train_SmolVLM.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   Training SmolVLM. The model is fine-tuned using a set of 1000 images with known values. The training process (12 epochs) takes about 3 hours on a single 80Gb H100 GPU.

.. figure:: illustrations/SmolVLM_raw_v_trained_example.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   Example results from SmolVLM. Original in the centre, after training on the right. Text in black is where the model is correct, in red shows where it is wrong (and where it is wrong, the correct value is given underneath in blue).

.. figure:: illustrations/SmolVLM_trained_stats.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   SmolVLM percentage success rate for each position in the table. After training.

An ensemble of trained models
-----------------------------

.. figure:: illustrations/SmolVLM_Granite_Gemma_trained.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   We can do the same fine-tuneing, on the same training set, on all three models.


.. figure:: illustrations/SmolVLM_Granite_Gemma_trained_example.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   Example results from the three-model ensemble after training. Values in blue show where 2 or more models agree on a value, and that value is correct. Values in red show where 2 or more models agree on a value, but that value is wrong. Values in grey show where there is no agreement among the models.

.. figure:: illustrations/SmolVLM_Granite_Gemma_trained_stats.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   Percentage success rate for each position in the table, for the three model ensemble after training. Value in blue is the percentage of values where 2 or more models agree, and that value is correct. Value in red is the percentage of values where 2 or more models agree, but that value is wrong. 


Conclusions
-----------

.. figure:: illustrations/trained_celebrating.png
   :width: 95%
   :align: center
   :figwidth: 95%


.. figure:: illustrations/T_shirt.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   Be willing to try something new - AIs may seem a bit strange, but still be very powerful


.. toctree::
   :maxdepth: 1

   How to reproduce or extend this work <how_to>
   Authors and acknowledgements <credits>

This document is distributed under the terms of the `Open Government Licence <https://www.nationalarchives.gov.uk/doc/open-government-licence/version/2/>`_. Source code included is distributed under the terms of the `BSD licence <https://opensource.org/licenses/BSD-2-Clause>`_.

