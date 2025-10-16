SmolVLM: Extract data from a set of images
==========================================

Same as :doc:`extract.py <extract>`, except that it does multiple images.

You can't specify a set of images to be extracted, the number you select are generated randomly, except that you can set the random number generator seed, so you can get the same set on multiple calls.

.. literalinclude:: ../../../models/smolvlm/extract_multi.py

    