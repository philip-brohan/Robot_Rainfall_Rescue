Gemma-3-4b: Extract data from a set of images
=============================================

Much the same as the :doc:`script to run SmolVLM <../smolvlm/extract_multi>`. The main difference is that Gemma requires image input to be pre-cut into squares - so one rectangular page has to be explicitly subdivided). I don't know how best to do the subdivision, but the default used by this scruipt works OK - at least after fine-tuneing.


.. literalinclude:: ../../../models/gemma/extract_multi.py

    