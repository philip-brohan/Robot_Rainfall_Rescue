Multi-model agreement: statistics of set of images
==================================================

.. figure:: stats_agreement.webp
   :width: 65%
   :align: center

   Percentage success rate for each position in the table, for the three model ensemble. Value in blue is the percentage of values where 2 or more models agree, and that value is correct. Value in red is the percentage of values where 2 or more models agree, but that value is wrong.

   As with the :doc:`single model statistics <plot_stats>`, it will compare statistics from all images for which extracted values are available for all models.

.. literalinclude:: ../../../models/gemma/plot_stats_agreement.py

    