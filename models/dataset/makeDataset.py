# Make tf.data.Datasets from the ATB2 fake data

import os
import tensorflow as tf
import numpy

# Load an image tensor from a file
def load_image_tensor(file_name):
    sict = tf.io.read_file(file_name)
    imt = tf.io.parse_tensor(sict, numpy.float32)
    # Threshold
    imt=tf.where(
        tf.less(imt, tf.zeros_like(imt) + 0.75),
        tf.zeros_like(imt),
        tf.ones_like(imt)
    )
    imt = tf.reshape(imt, [1024, 640, 1])
    return imt


# Load an standardised image tensor from a file
def load_standardised_tensor(file_name):
    sict = tf.io.read_file(file_name)
    imt = tf.io.parse_tensor(sict, numpy.float32)
    imt = tf.reshape(imt, [512, 768, 3])
    return imt


# Load an numbers tensor from a file
def load_numbers_tensor(file_name):
    sict = tf.io.read_file(file_name)
    imt = tf.io.parse_tensor(sict, numpy.float32)
    imt = tf.reshape(imt, [436, 10])
    return imt


# Load a corners tensor from a file
def load_corners_tensor(file_name):
    sict = tf.io.read_file(file_name)
    imt = tf.io.parse_tensor(sict, numpy.float32)
    #imt = imt[0:4]
    imt = tf.reshape(imt, [44])
    return imt


# Load a grid tensor from a file
def load_grid_tensor(file_name):
    sict = tf.io.read_file(file_name)
    imt = tf.io.parse_tensor(sict, numpy.float32)
    imt = tf.reshape(imt, [240])
    return imt


load_functions = {
    "corners": load_corners_tensor,
    "cell-centres": load_grid_tensor,
    "numbers": load_numbers_tensor,
    "images": load_image_tensor,
    "standardised": load_standardised_tensor,
}


def dirBase(subdir):
    if subdir is None:
        return "%s/ML_ten_year_rainfall/training_data" % os.getenv("SCRATCH")
    else:
        return "%s/ML_ten_year_rainfall/training_data/%s" % (
            os.getenv("SCRATCH"),
            subdir,
        )


# Get a dataset - images, numbers, corners, or standardised images
def getDataset(group, purpose, selection=None, nImages=None, subdir=None):

    supported = ("images", "numbers", "corners", "cell-centres", "standardised")
    if group not in supported:
        raise ValueError("group must be one of %r." % supported)

    # Get a list of filenames containing image tensors
    inFiles = os.listdir("%s/tensors/%s" % (dirBase(subdir), group))

    # do we want the training set, the test set, or a single sample
    splitI = int(len(inFiles) * 0.9)
    if purpose == "training":
        inFiles = inFiles[:splitI]
    if purpose == "test":
        inFiles = inFiles[splitI:]
    if purpose == "sample":
        if selection is None or selection > len(inFiles):
            raise ValueError("Not enough files to get selection")
        inFiles = [inFiles[selection]]

    if nImages is not None:
        if len(inFiles) >= nImages:
            inFiles = inFiles[0:nImages]
        else:
            raise ValueError(
                "Only %d images available, can't provide %d" % (len(inFiles), nImages)
            )

    # Create TensorFlow Dataset object from the file namelist
    inFiles = ["%s/tensors/%s/%s" % (dirBase(subdir), group, x) for x in inFiles]
    tr_data = tf.data.Dataset.from_tensor_slices(tf.constant(inFiles))

    # Convert the Dataset from file names to file contents
    tr_data = tr_data.map(
        load_functions[group], num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # Optimisation
    tr_data = tr_data.cache()
    tr_data = tr_data.prefetch(tf.data.experimental.AUTOTUNE)

    return tr_data


def getImageDataset(purpose="training", selection=None, nImages=None, subdir=None):

    return getDataset(
        "images", purpose, selection=selection, nImages=nImages, subdir=subdir
    )


def getNumbersDataset(purpose="training", selection=None, nImages=None, subdir=None):

    return getDataset(
        "numbers", purpose, selection=selection, nImages=nImages, subdir=subdir
    )


def getCornersDataset(purpose="training", selection=None, nImages=None, subdir=None):

    return getDataset(
        "corners", purpose, selection=selection, nImages=nImages, subdir=subdir
    )


def getGridDataset(purpose="training", selection=None, nImages=None, subdir=None):

    return getDataset(
        "cell-centres", purpose, selection=selection, nImages=nImages, subdir=subdir
    )


def getStandardisedDataset(
    purpose="training", selection=None, nImages=None, subdir=None
):

    return getDataset(
        "standardised", purpose, selection=selection, nImages=nImages, subdir=subdir
    )
