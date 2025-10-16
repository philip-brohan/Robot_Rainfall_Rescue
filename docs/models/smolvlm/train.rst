SmolVLM: Train against a sample of images with transcriptions
-------------------------------------------------------------

Fine tunes SmolVLM against a sample of images with transcriptions.

* --model_id specifies the start model. Must be HuggingFaceTB/SmolVLM-Instruct or a fine-tuned derivative
* --run_id gives the name of the resulting model after fine-tuneing
* --nmax is the number of images to use in training. This number will be taken from the training set at random
* --random_seed is the random seed to use when picking images. Let's you specify the same set of images again
* --epochs is the number of epochs to train for. More epochs will take longer but may give better results

Many more parameters are specified in the script.

.. literalinclude:: ../../../models/smolvlm/train.py

