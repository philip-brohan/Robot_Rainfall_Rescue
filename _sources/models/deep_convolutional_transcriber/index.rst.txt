A deep convolutional transcriber
================================

As a first attempt at a model which might do the transcription, I'm using a straightforward deep convolutional approach: Six 2x2 strided convolutional layers to reduce the dimension of the image, a single fully connected layer to extract the transcribed numbers from the convolutional features, and a softmax layer to express the predicted transcriptions as probabilities for each digit. With ELU activations, and dropout in each layer to resist overfitting.

.. figure:: DCT.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Schematic of the deep-convolutional transcriber model

.. toctree::
   :maxdepth: 1

   Model specification <transcriberModel>
   Model fit script <training>

This model is not in any way optimised to match the problem - it's a first attempt using standard methods. It's not perfect, but, after 200 epochs of training, it has considerable skill:

.. toctree::
   :maxdepth: 1

   Example image with transcriptions <compare>
   Transcription accuracy by digit <pvp>
   Transcription accuracy by page location <place>




