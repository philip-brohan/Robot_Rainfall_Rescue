Script to convert the datum locations on all images to tf.tensors
=================================================================

Convert all the locations (note that this script does not run the conversion - it outputs a list of commands to be run - run those efficiently in parallel).

.. literalinclude:: ../../../training_data/whole_image/to_tensors/all_grid-points_to_tensor.py

