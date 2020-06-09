#!/usr/bin/env python

# Convolutional autoencoder for ten-year rainfall images.

import os
import sys
import tensorflow as tf
import tensorflow.keras.backend as K
import pickle
import numpy

# Load the model specification
from autoencoderModel import autoencoderModel

# Load the data source provider
from makeDataset import getImageDataset

# How many images to use?
nTrainingImages = 31823  # Max is 31823
nTestImages = 353  # Max is 3535

# How many epochs to train for
nEpochs = 200
# Length of an epoch - if None, same as nTrainingImages
nImagesInEpoch = 1000

if nImagesInEpoch is None:
    nImagesInEpoch = nTrainingImages

# Dataset parameters
bufferSize = 100  # Shouldn't make much difference
batchSize = 10  # Bigger is faster, but takes more memory, and probably is less accurate

# Set up the training data
trainingData = getImageDataset(purpose="training", nImages=nTrainingImages).repeat()
trainingData = tf.data.Dataset.zip((trainingData, trainingData))
trainingData = trainingData.shuffle(bufferSize).batch(batchSize)

# Set up the test data
testData = getImageDataset(purpose="test", nImages=nTestImages).repeat()
testData = tf.data.Dataset.zip((testData, testData))
testData = testData.batch(batchSize)

# Instantiate the model
autoencoder = autoencoderModel()

# Save the model weights and the history state after every epoch
history = {}
history["loss"] = []
history["val_loss"] = []


class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        save_dir = ("%s/ML_ten_year_rainfall/autoencoder/" + "Epoch_%04d") % (
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


class ReduceNoise(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        K.set_value(
            self.model.noiseStdDev, self.model.noiseStdDev * 0.9,
        )


# Train the autoencoder
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adadelta(
        learning_rate=1.0, rho=0.95, epsilon=1e-07, name="Adadelta"
    ),
    loss="mean_squared_error",
)
history = autoencoder.fit(
    x=trainingData,
    epochs=nEpochs,
    steps_per_epoch=nImagesInEpoch // batchSize,
    validation_data=testData,
    validation_steps=nTestImages // batchSize,
    verbose=1,
    callbacks=[ReduceNoise(), CustomSaver()],
)
