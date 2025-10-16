Robot Rainfall Rescue
=====================

The `Rainfall Rescue project <https://climatelabbook.substack.com/p/rainfall-rescue-5-years-on>`_ used contributions from 16,000 volunteers to rescue more than 5 million historical weather observations from paper records. The project was a great success, but it's proved hard to replicate and scale up - recruiting and managing volunteer contributions at scale is very challenging. So we'd like to `do data rescue with Artificial Intelligence (AI) <https://brohan.org/AI_daily_precip/>`_, instead of using volunteers, to get a process we can run at scale, and on demand.

Here I show that we can replicate the success of Rainfall Rescue using a few small `Vision Language Models (VLMs) <https://huggingface.co/blog/vlms>`_ instead of thousands of human volunteers. After fine-tuneing on 1000 images (1.5% of the full dataset), each VLM can recover about 95% of the rainfall records correctly. Using an ensemble of three fine-tuned VLMs, and requiring agreement between at least two models, we can recover about 98% of the records correctly. This is a similar accuracy rate to that achieved with human volunteers, and not only allows us to save the time of all the volunteers, but also the time of the project team in recruiting, training, and managing the project.


The problem
-----------

We aim to rescue the monthly station rainfall records in the `UK 10-year rainfall sheets <https://digital.nmla.metoffice.gov.uk/SO_d383374a-91c3-4a7b-ba96-41b81cfb9d67/>`_. Each sheet contains the monthly rainfall totals for a single station, for 10 years. Despite their overall similarity, there is enough variation between the sheets that a fully automated approach is challenging. Here are three sample images:


.. figure:: illustrations/Image_samples.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   Three sample images showing monthly average rainfall observations. Each image contains the data from one station for 10 years.

We have 66,000 JPEG images, success means converting each image into a table of numbers that can be ingested into a database. In practice I'm converting each image into structured text - a JSON file. The point of the project is not to rescue the data (that's already been done) but to demonstrate that we can do the rescue using AI. Having the data already rescued allows us to test how well the AI is doing, by comparing the AI results with the known values. It also allows us to train the AI, by showing it some images with known values, and asking it to learn how to extract the values from the images.

The first step is to get the sheet images, and the known values, from the original Rainfall Rescue project:

.. toctree::
   :maxdepth: 1

   Download the Rainfall Rescue data <rainfall_rescue/get_data>
   Diagnostic plots for Rainfall_Rescue data <rainfall_rescue/diagnostics>
   Utility functions <rainfall_rescue/utilities>
   Image utilities <RR_utils/image>

Meet the team
-------------

To convert images into JSON, we need an AI that can take images and text instructions as input, and produce text as output - a `Vision Language Model (VLM) <https://huggingface.co/blog/vlms>`_. There are already many such models to choose from. We are not using the well-known flagship AI models like `ChatGPT <https://chatgpt.com/>`_ or `Gemini <https://gemini.google.com/app>`_ because they are too difficult to fine-tune. Instead we are using smaller, open-weight models that can be run on a single (`H100 <https://www.nvidia.com/en-gb/data-center/h100/>`_) GPU. I picked three small VLMs arbitrarily from those available on `Huggingface <https://huggingface.co/>`_:

.. figure:: illustrations/smolvlm_granite_gemma.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Three small VLMs, `SmolVLM <https://huggingface.co/blog/smolvlm>`_, `Granite <https://www.ibm.com/granite>`_, and `Gemma <https://deepmind.google/models/gemma/gemma-3/>`_. Note that these depictions are not official images from the model developers, but `anthropomorphic representations created by our artist <https://chatgpt.com/share/68b5a6ff-fc00-8013-b62f-2c4032700aa6>`_.

`Huggingface <https://huggingface.co/>`_ has been invaluable to this project, not only providing access to the models, but also the tools to run and fine-tune them (`transformers <https://huggingface.co/docs/transformers/en/index>`_), and documentation and example code.

.. toctree::
   :maxdepth: 1

   Huggingface interface functions <RR_utils/hf>


A single small VLM: SmolVLM
---------------------------

We start with a single model, `SmolVLM <https://huggingface.co/blog/smolvlm>`_, which is a relatively small (2 billion parameter) open-weight model, designed to run on a single GPU (needs only 5Gb GPU RAM). We start by using the model out of the box, with no fine-tuning. The model is given an image, and a prompt asking it to extract the values from the table in the image.

.. toctree::
   :maxdepth: 1

   Code to use SmolVLM to extract data from a single image <models/smolvlm/extract>
   Prompts used to control the extraction <models/smolvlm/prompts>
   Code to plot the results <models/gemma/plot_image+extracted>

The model does have some skill, but is far from perfect. Here are example results on a single image:

.. figure:: illustrations/SmolVLM_raw_easy.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   SmolVLM performance on a single image. Text in black is where the model is correct, in red shows where it is wrong (and where it is wrong, the correct value is given underneath in blue). This is a relatively easy example, with no missing data.

So it's promising at the start of the image (top left), but degrades as it goes on, and makes a lot of mistakes by the end of the image (bottom right). And that was an easy image - here is a more challenging example, with some missing data:

.. figure:: illustrations/SmolVLM_raw_hard.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   SmolVLM performance on a single image. Text in black is where the model is correct, in red shows where it is wrong (and where it is wrong, the correct value is given underneath in blue). This is a challenging example, with missing data.

And we can generate statistics for its overall skill, by running it on a set of 100 images, and computing the percentage success for each position in the data table.

.. toctree::
   :maxdepth: 1

   Code to run SmolVLM on a set of images <models/smolvlm/extract_multi>
   Plot percentage success rate <models/gemma/plot_stats>
   

.. figure:: illustrations/SmolVLM_raw_stats.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   SmolVLM percentage success rate for each position in the table.

This is clearly nowhere near good enough - human volunteers would be about 97% accurate for each entry, and we are aiming to match that. So what can we do to improve the results?

An ensemble of VLMs
-------------------

If your model has some skill, but not enough, try some other models. We can do exactly the same analysis with two other small open-weight VLMs, `Google's Gemma <https://deepmind.google/models/gemma/gemma-3/>`_ and `IBM's Granite <https://www.ibm.com/granite>`_:

.. toctree::
   :maxdepth: 1

   Code to repeat the analysis with Google Gemma-3-4b <models/gemma/gemma>
   Code to repeat the analysis with IBM Granite <models/granite/granite>

Gemma and Granite have the same fundamental issues as SmolVLM, each is better in some respects, but none is good enough to use alone. But what if we combine them? An ensemble of models often outperforms any individual model - the ensemble mean is an improved best estimate, and the ensemble spread is a useful measure of uncertainty. We won't do exactly that here, because we are looking for categorical values (the numbers in the table), so we don't want a mean, but we can look for agreement between models. What if we look for cases where two or more models agree on a value?

.. toctree::
   :maxdepth: 1

   Code to plot multi-model results compared with a sample image <models/gemma/plot_agreement>

.. figure:: illustrations/SmolVLM_Granite_Gemma_raw_example.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   Example results from the three-model ensemble (untrained). Values in blue show where 2 or more models agree on a value, and that value is correct. Values in red show where 2 or more models agree on a value, but that value is wrong. Values in grey show where there is no agreement among the models.

This is encouraging - for the rainfall data entries, about half the time we have multi-model agreement, and when we do, they are always right - the process is not overconfident. But that is just one image (and an easy one with no missing data). We can repeat the process over a set of 100 images (as before), and generate statistics:

.. toctree::
   :maxdepth: 1

   Code to plot multi-model statistics over a set of images <models/gemma/plot_stats_agreement>


.. figure:: illustrations/SmolVLM_Granite_Gemma_raw_stats.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   Percentage success rate for each position in the table, for the three model ensemble. Value in blue is the percentage of values where 2 or more models agree, and that value is correct. Value in red is the percentage of values where 2 or more models agree, but that value is wrong. 

The ensemble of untrained models successfully recovers about half of the rainfall values. This would be great - 50% of the data is a great deal better than nothing - but unfortunately, the process is somewhat overconfident - about 5-10% of the time 2 or more models agree on an incorrect value (there's some kind of common-mode failure between the models). We need to reduce the false success rate, and increase the overall success rate. It's time to do some fine-tuneing.

Training the models
-------------------

Let's divide our problem set of 66,000 images into two groups: a training set of 1000 images (1.5% of the total), and a test set of 65,000 images (the remaining 98.5%). We will use the training set to fine-tune the models, and then test the results on the test set. The cost of this is that we have to transcribe the values in the training set by some other process - probably manual data entry, possibly with human volunteers. But 1000 images is a lot less work than 66,000 images, we're still planning to do 98.5% of the work with AI.

And, of course, because we are working with images with known values, we can go straight to the training. Let's start with SmolVLM:

.. figure:: illustrations/Train_SmolVLM.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   Training SmolVLM. The model is fine-tuned using a set of 1000 images with known values. The training process (12 epochs) takes about 3 hours on a single 80Gb H100 GPU.

.. toctree::
   :maxdepth: 1

   Code to fine-tune SmolVLM <models/smolvlm/train>

Because we are working with a small model, and a small training set, the fine-tuning is quick and easy - the cost is negligible. And we can see how well it works by comparing the output of the trained model, with that of the untrained model, on a sample image from the test set:

.. toctree::
   :maxdepth: 1

   Run the trained model on a single image <models/smolvlm/extract>
   Show comparison between two models for a single image <models/gemma/plot_image+comparison>

This works very well - a single image shows very promising:

.. figure:: illustrations/SmolVLM_raw_v_trained_example.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   Example results from SmolVLM. Original in the centre, after training on the right. Text in black is where the model is correct, in red shows where it is wrong (and where it is wrong, the correct value is given underneath in blue).

And we can generate statistics for the trained model in exactly the same way as for the untrained model, by running it on a set of 100 images, and computing the percentage success for each position in the data table.

.. figure:: illustrations/SmolVLM_trained_stats.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   SmolVLM percentage success rate for each position in the table. After training.

Trained SmolVLM is not quite as good as a human volunteer, but it's close - consistently getting more than 90% of rainfall values right. That's still not quite good enough on its own, but what if we do the same training on the other two models, and then use the ensemble approach again?

An ensemble of trained models
-----------------------------

.. figure:: illustrations/SmolVLM_Granite_Gemma_trained.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   We can do the same fine-tuneing, on the same training set, on all three models.

.. toctree::
   :maxdepth: 1

   Code to train Gemma <models/gemma/train>
   Code to train Granite <models/granite/train>

Gemma and Granite are bigger models, so the training takes longer (about 12 hours on a single 80Gb H100 GPU), but the cost is still tiny. And we can look at the output of the trained ensemble, exactly as we did with the untrained ensemble above, first on a single test image:

.. figure:: illustrations/SmolVLM_Granite_Gemma_trained_example.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   Example results from the three-model ensemble after training. Values in blue show where two or more models agree on a value, and that value is correct. Values in red show where two or more models agree on a value, but that value is wrong. Values in grey show where there is no agreement among the models.

.. figure:: illustrations/SmolVLM_Granite_Gemma_trained_stats.jpg
   :width: 95%
   :align: center
   :figwidth: 95%

   Percentage success rate for each position in the table, for the three model ensemble after training. Value in blue is the percentage of values where two or more models agree, and that value is correct. Value in red is the percentage of values where two or more models agree, but that value is wrong. 

This works *very* well - the trained ensemble is not perfect, but it is very good - consistently in the high nineties and the false success rate is down to about 1%. This is comparable to the performance of citizen scientists, but without the need to recruit, train, and manage thousands of volunteers.

In an ideal world, we'd get the success rate up to 100%, and the false success rate down to 0. And with more training, or a larger set of VLMs, or better prompts, we could get closer to that. But it's probably not worth it - even with perfect transcription the recovered rainfall values would still contain errors, because the original observations contain errors - realistically, this is already good enough.

Conclusions
-----------

An ensemble of three small VLMs, fine-tuned on a training set of 1000 images, can recover about 98% of the rainfall values correctly, with a false success rate of about 1%. There is no need for any task-specific processing - the AIs can do 100% of the job.

.. figure:: illustrations/trained_celebrating.png
   :width: 95%
   :align: center
   :figwidth: 95%

This approach is cheap and easy and effective. The next step is to use it in anger - to apply it to a `large set of images that have *not* yet been rescued <https://digital.nmla.metoffice.gov.uk/index.php?name=SO_9903efdf-7f99-4cae-a723-8b3f426eea20>`_.

My other main takeaway from this project is that I'm too `square <https://www.urbandictionary.com/define.php?term=square>`_. I was very slow to appreciate `Huggingface <https://huggingface.co/>`_ - how can you take seriously a web-site named after an emoji (🤗)? And I was slow to internalize the `bitter lesson <http://www.incompleteideas.net/IncIdeas/BitterLesson.html>`_ - don't try and impose your own structure on the problem, just deploy generic AI and let it learn. Use pre-trained general-purpose AIs and use them on as much of the problem as possible.

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

