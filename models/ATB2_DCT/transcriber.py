#!/usr/bin/env python

# Transcriber for simplified transfer-learning images.

import os
import sys
import tensorflow as tf
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

# Load the data source providers
sys.path.append("%s/../dataset" % os.path.dirname(__file__))
from makeDataset import getImageDataset
from makeDataset import getNumbersDataset

# Load the model specification
from transcriberModel import transcriberModel

# How many images to use?
nTrainingImages = 9000  # Max is 9000
nTestImages = 1000  # Max is 1000

# How many epochs to train for
nEpochs = 200
# Length of an epoch - if None, same as nTrainingImages
nImagesInEpoch = 1000

if nImagesInEpoch is None:
    nImagesInEpoch = nTrainingImages

# Dataset parameters
bufferSize = 100  # Shouldn't make much difference (data is random)
batchSize = 32  # Arbitrary

# Set up the training data
imageData = getImageDataset(
    subdir="unperturbed", purpose="training", nImages=nTrainingImages
).repeat()
numbersData = getNumbersDataset(
    subdir="unperturbed", purpose="training", nImages=nTrainingImages
).repeat()
trainingData = tf.data.Dataset.zip((imageData, numbersData))
trainingData = trainingData.shuffle(bufferSize).batch(batchSize)

# Set up the test data
testImageData = getImageDataset(
    subdir="unperturbed", purpose="test", nImages=nTestImages
).repeat()
testNumbersData = getNumbersDataset(
    subdir="unperturbed", purpose="test", nImages=nTestImages
).repeat()
testData = tf.data.Dataset.zip((testImageData, testNumbersData))
testData = testData.batch(batchSize)

# Instantiate the model
with strategy.scope():
    transcriber = transcriberModel()
    transcriber.compile(
        optimizer=tf.keras.optimizers.Adadelta(
            learning_rate=1.0, rho=0.95, epsilon=1e-07, name="Adadelta"
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(),
    )
    # If we are doing a restart, load the weights
    if args.epoch > 0:
        weights_dir = ("%s/Robot_Rainfall_Rescue/models/ATB2_DCT/Epoch_%04d") % (
            os.getenv("SCRATCH"),
            args.epoch - 1,
        )
        load_status = seeker.load_weights("%s/ckpt" % weights_dir)
        # Check the load worked
        load_status.assert_existing_objects_matched()

# Save the model weights and the history state after every epoch
history = {}
history["loss"] = []
history["val_loss"] = []


class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        save_dir = ("%s/Robot_Rainfall_Rescue/models/ATB2_DCT/Epoch_%04d") % (
            os.getenv("SCRATCH"),
            epoch,
        )
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.model.save_weights("%s/ckpt" % save_dir)
        history["loss"].append(logs["loss"])
        history["val_loss"].append(logs["val_loss"])
        history_file = "%s/history.pkl" % save_dir
        pickle.dump(history, open(history_file, "wb"))


# Train the transcriber
transcriber.compile(
    optimizer=tf.keras.optimizers.Adadelta(
        learning_rate=1.0, rho=0.95, epsilon=1e-07, name="Adadelta"
    ),
    loss=tf.keras.losses.CategoricalCrossentropy(),
)
history = transcriber.fit(
    x=trainingData,
    epochs=nEpochs,
    initial_epoch=args.epoch,
    steps_per_epoch=nImagesInEpoch // batchSize,
    validation_data=testData,
    validation_steps=nTestImages // batchSize,
    verbose=1,
    callbacks=[CustomSaver()],
)
