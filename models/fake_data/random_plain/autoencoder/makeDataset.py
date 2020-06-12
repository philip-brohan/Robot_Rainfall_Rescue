# Make a tf.data.Dataset from the random plain fake image tensors

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
    inFiles = os.listdir(
        "%s/ML_ten_year_rainfall/fakes/plain/tensors/images" % os.getenv("SCRATCH")
    )
    splitI = int(len(inFiles) * 0.9)
    if purpose == "training":
        inFiles = inFiles[:splitI]
    if purpose == "test":
        inFiles = inFiles[splitI:]

    if nImages is not None:
        if len(inFiles) >= nImages:
            inFiles = inFiles[0:nImages]
        else:
            raise ValueError(
                "Only %d images available, can't provide %d" % (len(inFiles), nImages)
            )

    # Create TensorFlow Dataset object from the file namelist
    inFiles = [
        "%s/ML_ten_year_rainfall/fakes/plain/tensors/images/%s"
        % (os.getenv("SCRATCH"), x)
        for x in inFiles
    ]
    itList = tf.random.shuffle(tf.constant(inFiles))
    tr_data = tf.data.Dataset.from_tensor_slices(itList)

    # Convert the Dataset from file names to file contents
    tr_data = tr_data.map(load_tensor,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Optimisation
    tr_data = tr_data.prefetch(tf.data.experimental.AUTOTUNE)

    return tr_data
