Plot success percentages from a model
=====================================

.. figure:: stats.webp
   :width: 65%
   :align: center

   Success percentages for the Gemma-3-4B-IT model.


You don't specify which images to use - it will use all available images from either the test (default) or training sets for which extractions are available.
You will need to do the :doc:`extractions <extract_multi>` first.

.. literalinclude:: ../../../models/gemma/plot_stats.py


