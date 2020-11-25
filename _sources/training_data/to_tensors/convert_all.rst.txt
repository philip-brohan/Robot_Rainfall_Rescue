Script to convert all the transfer-learning data to tf.tensors
==============================================================

Convert all the images (note that this script does not run the conversion - it outputs a list of commands to be run - run those efficiently in parallel).

.. literalinclude:: ../../../training_data/whole_image/to_tensors/all_images_to_tensor.py

Convert all the number arrays (note that this script does not run the conversion - it outputs a list of commands to be run - run those efficiently in parallel).

.. literalinclude:: ../../../training_data/whole_image/to_tensors/all_numbers_to_tensor.py
