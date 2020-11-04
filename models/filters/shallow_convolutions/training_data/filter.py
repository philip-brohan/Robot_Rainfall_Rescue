#!/usr/bin/env python

# Convolutional filter for ten-year rainfall training images.

import os
import sys
import tensorflow as tf
import tensorflow.keras.backend as K
import pickle
import numpy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--epoch", help="Restart from epoch", type=int, required=False, default=0
)
args = parser.parse_args()

# Distribute across all GPUs
strategy = tf.distribute.MirroredStrategy()

# Load the model specification
sys.path.append("%s/.." % os.path.dirname(__file__))
from filterModel import filterModel

# Load the data source provider
sys.path.append("%s/../../../dataset" % os.path.dirname(__file__))
from makeDataset import getImageDataset

# How many images to use?
nTrainingImages = 9000  # Max is 9000
nTestImages = 1000  # Max is 1000

# How many epochs to train for
nEpochs = 500
# Length of an epoch - if None, same as nTrainingImages
nImagesInEpoch = 9000

if nImagesInEpoch is None:
    nImagesInEpoch = nTrainingImages

# Dataset parameters
bufferSize = 1000  # Shouldn't make much difference
batchSize = 32  # Arbitrary

# Set up the training data
trainingData = getImageDataset(purpose="training", nImages=nTrainingImages).repeat()
trainingData = tf.data.Dataset.zip((trainingData, trainingData))
trainingData = trainingData.shuffle(bufferSize).batch(batchSize)

# Set up the test data
testData = getImageDataset(purpose="test", nImages=nTestImages).repeat()
testData = tf.data.Dataset.zip((testData, testData))
testData = testData.batch(batchSize)

# Instantiate the model
with strategy.scope():
    filterM = filterModel()
    filterM.compile(
        optimizer=tf.keras.optimizers.Adadelta(
            learning_rate=1.0, rho=0.95, epsilon=1e-07, name="Adadelta"
        ),
        loss="mean_squared_error",
    )
    # If we are doing a restart, load the weights
    if args.epoch > 0:
        weights_dir = (
            "%s/ML_ten_year_rainfall/models/filter/training/" + "Epoch_%04d"
        ) % (os.getenv("SCRATCH"), args.epoch - 1,)
        load_status = filterM.load_weights("%s/ckpt" % weights_dir)
        # Check the load worked
        load_status.assert_existing_objects_matched()


# Save the model weights and the history state after every epoch
history = {}
history["loss"] = []
history["val_loss"] = []


class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        save_dir = (
            "%s/ML_ten_year_rainfall/models/filter/training/" + "Epoch_%04d"
        ) % (os.getenv("SCRATCH"), epoch,)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.model.save_weights("%s/ckpt" % save_dir)
        history["loss"].append(logs["loss"])
        history["val_loss"].append(logs["val_loss"])
        history_file = "%s/history.pkl" % save_dir
        pickle.dump(history, open(history_file, "wb"))

# Train the filter
history = filterM.fit(
    x=trainingData,
    epochs=nEpochs,
    steps_per_epoch=nImagesInEpoch // batchSize,
    validation_data=testData,
    validation_steps=nTestImages // batchSize,
    verbose=1,
    callbacks=[CustomSaver()],
)
