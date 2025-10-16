Multi-model agreement: single image
====================================

.. figure:: agree.webp
   :width: 95%
   :align: center

   Example results from the three-model ensemble (untrained). Values in blue show where 2 or more models agree on a value, and that value is correct. Values in red show where 2 or more models agree on a value, but that value is wrong. Values in grey show where there is no agreement among the models.

.. literalinclude:: ../../../models/gemma/plot_agreement.py

    