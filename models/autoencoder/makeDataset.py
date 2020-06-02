# Make a tf.data.Dataset from the ten-year rainfall image tensors

import os
import tensorflow as tf
import numpy

# Load an image tensor from a file
def load_tensor(file_name):
    sict = tf.io.read_file(file_name)
    imt = tf.io.parse_tensor(sict, numpy.float32)
    imt = tf.reshape(imt, [1024, 768, 3])  # Already this shape
    return imt


# Get a logbook tensors dataset - for 'training' or 'test'.
#  Optionally specify how many images to use.
def getImageDataset(purpose="training", nImages=None):

    # Get a list of filenames containing image tensors
    prds = os.listdir(
        "%s/ML_ten_year_rainfall/images/%s" % (os.getenv("SCRATCH"), purpose)
    )
    inFiles = []
    for prd in prds:
        dirs = os.listdir(
            "%s/ML_ten_year_rainfall/images/%s/%s"
            % (os.getenv("SCRATCH"), purpose, prd)
        )
        for dirn in dirs:
            files = os.listdir(
                "%s/ML_ten_year_rainfall/images/%s/%s/%s"
                % (os.getenv("SCRATCH"), purpose, prd, dirn)
            )
            for filen in files:
                inFiles.append(
                    "%s/ML_ten_year_rainfall/images/%s/%s/%s/%s"
                    % (os.getenv("SCRATCH"), purpose, prd, dirn, filen)
                )
    if nImages is not None:
        if len(inFiles) >= nImages:
            inFiles = inFiles[0:nImages]
        else:
            raise ValueError(
                "Only %d images available, can't provide %d" % (len(inFiles), nImages)
            )

    # Create TensorFlow Dataset object from the file namelist
    itList = tf.constant(inFiles)
    tr_data = tf.data.Dataset.from_tensor_slices(itList)

    # Convert the Dataset from file names to file contents
    tr_data = tr_data.map(load_tensor)

    return tr_data
