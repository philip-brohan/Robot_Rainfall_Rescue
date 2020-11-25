Finding grid corners
====================

Finding :doc:`datum locations proved difficult <../find_grid/index>` - what if we try to find grid-cell corners instead? This might be easier, as we are looking for a more consistent mark on the page.

.. figure:: individual.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Grid-corner locations for one member of the transfer-learning dataset: true locations on the left, locations produced by an ML model from the image on the right.

This requires an extension to the :doc:`training dataset <../../training_data/itp>`: as well as the image tensors already in the dataset, we need, for each image,  a training target that is a tensor containing the page coordinates of each grid-corner:

.. toctree::
   :maxdepth: 1

   Script to make a tensor of the grid-points <../../training_data/to_tensors/corners_to_tensor>
   Script to do this 10,000 times (once for each training image)<../../training_data/to_tensors/all_corners>

Then we specify a deep convolutional model with the images as input and the grid-corner locations as output, and fit it as before. 

.. toctree::
   :maxdepth: 1

   Model specification <transcriberModel>
   Model fit script <training>

Then we need to see if it works. There are two validation checks needed: the first to see how it does on the validation subset of the transfer-learning dataset on which it has been trained, and the second (harder) to see how well its skill transfers across to the real dataset.

.. toctree::
   :maxdepth: 1

   Training-dataset validation <validation>
   Real-dataset cross-validation <cross-validation>

It works better than the :doc:`direct datum location search <../find_grid/index>`, but even on the training dataset precision is limited, and the skill does not transfer to the real dataset well. Again, more work needed.


