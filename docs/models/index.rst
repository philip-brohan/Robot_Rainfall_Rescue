Direct transcription of a simplified training dataset
=====================================================

First, let's find out how hard it is to train an ML model to transcribe the simplified training dataset: 

.. toctree::
   :maxdepth: 1

   A deep convolutional transcriber <deep_convolutional_transcriber/index>
   A tuned convolutional transcriber <tuned_convolutional_transcriber/index>

This is very encouraging: The tuned transcriber does the job essentially perfectly - it can take the images in the simplified training dataset, and recover the numbers from the images. This won't help with the real problem, as it will only work with numbers in a fixed font at a set location on the page (the simplified training case), but it does illustrate the power of machine learning: Writing a program to do the transcription (even in the simple case) would be very difficult - training an ML model to do the job is easy and fast.
 
