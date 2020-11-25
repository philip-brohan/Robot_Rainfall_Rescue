Functions that package all the transfer-learning tensors as tf.data.Datasets
============================================================================

Call these functions in the ML model fitting script - they output `TensorFlow Datasets <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_ which can be used directly in model fitting.

.. literalinclude:: ../../../models/dataset/makeDataset.py

