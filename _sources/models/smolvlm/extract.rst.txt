SmolVLM: Extract data from a single image
-----------------------------------------

This script uses SmolVLM to extract data from a single image.

* Argument ```--label``` specifies which image (from the :doc:`Rainfall Rescue collection <../../rainfall_rescue/get_data>`). Image labels are strings like ```TYRain_1920-1930_06_pt1-page-270```. If omitted, picks an image at random.
* Argument ```--model``` specifies which model to use. This script is for SmolVLM only but it works for original and trained versions. If omitted, uses the untrained version, to use a trained version, use the model name specified in the training script.

Output is put under $PDIR/extracted. (PDIR is set in the :doc:`environment <../../how_to>`).

.. literalinclude:: ../../../models/smolvlm/extract.py

