Robot Rainfall Rescue
=====================

Can we use Machine Learning to rapidly transcribe vital climate data from paper archives? 

Datasets of historical weather observations are vital to our understanding of climate change and variability, and improving those datasets means transcribing millions of observations - converting paper records into a digital form. Doing such transcription manually is `expensive and slow <http://brohan.org/transcription_methods_review/>`_, and we have a backlog of millions of pages of potentially valuable records which have never been transcribed. We would dearly like a cheap, fast, software tool for extracting weather observations from (photographs of) archived paper documents. No such system currently exists, but recent developments in machine learning methods and image analysis tools suggest that it might now be possible to create one.

This is an attempt to create such a tool: specifically it is an attempt to use the `TensorFlow <https://www.tensorflow.org/>`_ machine learning toolkit to reproduce the work done by 16,000 human volunteers in the `Rainfall Rescue project <https://www.zooniverse.org/projects/edh/rainfall-rescue>`_.

--------

The requirement is to extract weather observations from document images, producing machine-readable output.

.. figure:: The_problem.png
   :width: 95%
   :align: center
   :figwidth: 95%

   How can we do this: image to text conversion, 1,000,000 times, quickly and cheaply?

This transcription is easy for humans, but it's laborious and slow. We need a software solution. Optical Character Recognition (OCR) software does not work well, and is a poor tool for this task in any event - here we need not only to recognise the characters, but to find where on the page the data of interest are, and to preserve the structure of the data table in the output. We also need to cope with variations - even in a single data source, the document images are rarely all exactly the same:

.. figure:: rainfall_rescue_data/Variations.png
   :width: 95%
   :align: center
   :figwidth: 95%


So we need to create software that is powerful enough to find the grid and transcribe the digits from each image, flexible enough to cope with the many slightly-different image formats and colours, and adaptable enough that it can be re-purposed to transcribe records from other document types with different formats.

This is a staggeringly difficult task, but deep-learning methods have demonstrated remarkable capability on other difficult tasks in the general field of image analysis, can we use deep-learning for transcription?

--------

We're trying to come up with a general method. So we want a test dataset to work on that's fairly typical - neither too easy nor to difficult. We need a dataset where the answers are already available - so we can easily test to see how well the transcription tool is working. I've chosen the 10-year rainfall sheets: lose-leaf forms, each recording monthly rainfall at a single UK station over a period of 10-years. The `collection of these in the UK Meteorological Archive <https://digital.nmla.metoffice.gov.uk/SO_d383374a-91c3-4a7b-ba96-41b81cfb9d67/>`_ comprises about 65,000 such sheets covering 1677-1960 (though the early years include very few stations). These were manually digitised in spring 2020 by the `Rainfall Rescue citizen science project <https://www.zooniverse.org/projects/edh/rainfall-rescue>`_, so both the document images, and the transcriptions, are readily available.

.. toctree::
   :maxdepth: 1

   Getting the images to transcribe <rainfall_rescue_data/index>

--------

The software we need is a program that takes an image as input, and generates a table of transcribed numbers (observations) as output. `Machine Learning (ML) <https://en.wikipedia.org/wiki/Machine_learning>`_ is a process for automatically generating programs from paired examples of input and output. So as we have both images (input) and transcriptions (output) for the 10-year-rainfall dataset, we could directly train an ML model to do the transcription.

But this isn't quite what we want, because it requires us to already have the transcriptions to generate the model, and in general we need the model first to produce the transcriptions: We need to learn a transcription model despite not having any transcribed outputs to train on. The process for this is `Transfer Learning <https://en.wikipedia.org/wiki/Transfer_learning>`_ - we train our transcription model on a different, but related set of image::transcription pairs that we already have, and then apply that same model to transcribe the images we are interested in. We don't have a set of very similar image::transcription pairs to do the training on, but we can make them.

So instead of writing software to convert images to transcriptions (*very* hard), we write software to make images from transcriptions (much easier), use this to make a big dataset of transcription::image pairs, and then use this dataset to train an ML system to invert the process - to make transcriptions from images.

.. toctree::
   :maxdepth: 1

   Preparing a transfer dataset for training <training_data/itp>

---------


We now have a transfer dataset for training, and a real dataset for validation. So all we need to do is design an ML model architecture that is powerful enough to do the transcription, train the model on the transfer dataset, deploy the trained model on the real (validation) dataset, and verify that it works.

Unfortunately it's not clear `how to select the right model <https://en.wikipedia.org/wiki/Model_selection>`_, or what `hyperparameters <https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)>`_ to use in its training. So we are going to have to do some experiments - training a series of different models, on simplified problems, to find out what works.

.. toctree::
   :maxdepth: 2

   Direct transcription of the simplified training dataset <models/atb2>
   Modelling the page layout <models/layout>


---------

This project uses only open-source software and publicly-available data. It should be easy to replicate and improve upon. Can you train a better transcriber? Or do you want to try this method on a different image collection? If so, **please do**. 

.. toctree::
   :maxdepth: 1

   How to reproduce or extend this work <how_to>
   Authors and acknowledgements <credits>

This document is distributed under the terms of the `Open Government Licence <https://www.nationalarchives.gov.uk/doc/open-government-licence/version/2/>`_. Source code included is distributed under the terms of the `BSD licence <https://opensource.org/licenses/BSD-2-Clause>`_.

