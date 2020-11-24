Transfer-learning images and data
=================================

Images for transfer learning should contain the key feature - the data table - but we don't need to include all the other details.

.. figure:: examples/unperturbed/0000.png
   :width: 400
   :align: center
   :figwidth: 85%

   An example transfer-learning image

For each year (of 10), there are data for 12 months. The data for one month is three random digits (x.yz, where x, y, and z are random digits [0-9]). Then there are 12 sets of 3 digits for the monthly means, and 10 sets of 4 digits for the annual totals. So the associated data array is a 10x12x3+12x3+10x4 array, 436 digits.

.. literalinclude:: examples/unperturbed/0000.py

But this image is too specific to be a good transfer learning target, we need a range of image formats to cover the variation we see in the real images.

.. toctree::
   :maxdepth: 1

   Image variations <variations>

And we need scripts to make thousands of such image:data pairs for model training:

.. toctree::
   :maxdepth: 1

   Dataset scripts <scripts>

And we need scripts to package the transfer-learning training in the format required for use by TensorFlow:

.. toctree::
   :maxdepth: 1

   Package dataset for ML training <to_tensors>

