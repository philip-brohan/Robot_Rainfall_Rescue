Scripts to make the transfer-learning training dataset
======================================================

We need to make a set of 10,000 image::data pairs for ML model training. we do this with three scripts: A script to make a single image::data pair, a script to run the single pair generator 10,000 times with different random perturbations each time, and a class to abstract out the process of creating the image.

.. toctree::
   :maxdepth: 1

   scripts/image_class
   scripts/image_data_pair
   scripts/dataset

This process makes a training dataset with a lot of variability (varying fonts, text angle, position on the page, ...). This is good for transfer learning as we want a training dataset with more variability that the real images, so model skill will transfer over. It's not so good for model building though - it makes it more difficult to find ML models that work with the images.

So for experimentation it's useful to have another training dataset, this one with no variability (except for the digits in the image - only one font, always the same text angle and position on the page). Call this the simplified training dataset - we can make it with the same scripts as above, except for a modification to the overall dataset training script.

.. toctree::
   :maxdepth: 1

   scripts/dataset_simple



