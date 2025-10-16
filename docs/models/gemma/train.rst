Gemma-3-4b: Train against a sample of images with transcriptions
----------------------------------------------------------------

Much the same as the :doc:`script to fine tune SmolVLM <../smolvlm/train>`. The main difference is that Gemma requires image input to be pre-cut into squares - so one rectangular page has to be explicitly subdivided). I don't know how best to do the subdivision, but the default used by this script works OK - at least after fine-tuneing. (I have not checked, but pobably it's important to use the same subdivision method for training and for extraction).

.. literalinclude:: ../../../models/gemma/train.py

