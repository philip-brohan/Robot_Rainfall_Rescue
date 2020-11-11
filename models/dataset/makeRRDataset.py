# Make tf.data.Datasets from the rainfall rescue image tensors and numbers tensors

import os
import tensorflow as tf
import numpy

# Load an image tensor from a file
def load_image_tensor(file_name):
    sict = tf.io.read_file(file_name)
    imt = tf.io.parse_tensor(sict, numpy.float32)
    # Threshold: 0 if < 0.75, 1 otherwise
    imt = tf.where(
        tf.less(imt, tf.zeros_like(imt) + 0.75),
        tf.zeros_like(imt),
        tf.ones_like(imt)
    )
    imt = tf.reshape(imt, [1024, 640, 1])
    return imt


# Load an numbers tensor from a file
def load_numbers_tensor(file_name):
    sict = tf.io.read_file(file_name)
    imt = tf.io.parse_tensor(sict, numpy.float32)
    imt = tf.reshape(imt, [520, 11])
    return imt


# Get an image tensors dataset - for 'training' or 'test'.
#  Optionally specify how many images to use.
def getImageDataset(purpose="training", nImages=None):

    baseD = "%s/ML_ten_year_rainfall/tensors/images/%s" % (
        os.getenv("SCRATCH"),
        purpose,
    )

    # Get a list of filenames containing image tensors
    inFiles = sorted(os.listdir(baseD))

    if nImages is not None:
        if len(inFiles) >= nImages:
            inFiles = inFiles[0:nImages]
        else:
            raise ValueError(
                "Only %d images available, can't provide %d" % (len(inFiles), nImages)
            )

    # Create TensorFlow Dataset object from the file namelist
    inFiles = ["%s/%s" % (baseD, x) for x in inFiles]
    tr_data = tf.data.Dataset.from_tensor_slices(tf.constant(inFiles))

    # Convert the Dataset from file names to file contents
    tr_data = tr_data.map(
        load_image_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # Optimisation
    tr_data = tr_data.prefetch(tf.data.experimental.AUTOTUNE)

    return tr_data


# Get an numbers tensors dataset - for 'training' or 'test'.
#  Optionally specify how many images to use.
def getNumbersDataset(purpose="training", nImages=None):

    baseD = "%s/ML_ten_year_rainfall/tensors/numbers/%s" % (
        os.getenv("SCRATCH"),
        purpose,
    )

    # Get a list of filenames containing numbers tensors
    inFiles = sorted(os.listdir(baseD))

    if nImages is not None:
        if len(inFiles) >= nImages:
            inFiles = inFiles[0:nImages]
        else:
            raise ValueError(
                "Only %d numbers available, can't provide %d" % (len(inFiles), nImages)
            )

    # Create TensorFlow Dataset object from the file namelist
    inFiles = ["%s/%s" % (baseD, x) for x in inFiles]
    tr_data = tf.data.Dataset.from_tensor_slices(tf.constant(inFiles))

    # Convert the Dataset from file names to file contents
    tr_data = tr_data.map(
        load_numbers_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # Optimisation
    tr_data = tr_data.prefetch(tf.data.experimental.AUTOTUNE)

    return tr_data
