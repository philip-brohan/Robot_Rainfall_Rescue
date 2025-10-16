Gemma-3-4b: Extract data from a single image
--------------------------------------------

Much the same as the :doc:`script to run SmolVLM <../smolvlm/extract>`. The main difference is that Gemma requires image input to be pre-cut into squares - so one rectangular page has to be explicitly subdivided). I don't know how best to do the subdivision, but the default used by this script works OK - at least after fine-tuneing.

.. literalinclude:: ../../../models/gemma/extract_json.py

