#!/usr/bin/env python

# Deep Convolutional GAN for ten-year rainfall images.

import os
import sys
import time
import tensorflow as tf

# Load the model specification
from generatorModel import generatorModel, generatorLoss, generatorOptimizer
from discriminatorModel import (
    discriminatorModel,
    discriminatorLoss,
    discriminatorOptimizer,
)

# Load the data source provider - same as the autoencoder
sys.path.append("%s/../autoencoder" % os.path.dirname(__file__))
from makeDataset import getImageDataset

# How many epochs to train for
nEpochs = 500

# Latent space dimension
latentDim = 100

# How many images to use?
nTrainingImages = 9000  # Max is 9000

# Dataset parameters
bufferSize = 100  # Shouldn't make much difference
batchSize = 16  # Big enough for a variety of discriminator results

# Set up the training data
trainingData = getImageDataset(purpose="training", nImages=nTrainingImages).repeat()
trainingData = trainingData.shuffle(bufferSize).batch(batchSize)

# Instantiate the models
generator = generatorModel()
discriminator = discriminatorModel()

# Specify what to save at a checkpoint
checkpoint = tf.train.Checkpoint(
    generatorOptimizer=generatorOptimizer,
    discriminatorOptimizer=discriminatorOptimizer,
    generator=generator,
    discriminator=discriminator,
)


# Explicit training loop
@tf.function
def trainStep(images):
    noise = tf.random.normal([batchSize, latentDim])

    with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
        generatedImages = generator(noise, training=True)

        realOutput = discriminator(images, training=True)
        fakeOutput = discriminator(generatedImages, training=True)

        lossG = generatorLoss(fakeOutput)
        lossD = discriminatorLoss(realOutput, fakeOutput)

    gradientsOfGenerator = genTape.gradient(lossG, generator.trainable_variables)
    gradientsOfDiscriminator = discTape.gradient(
        lossD, discriminator.trainable_variables
    )

    generatorOptimizer.apply_gradients(
        zip(gradientsOfGenerator, generator.trainable_variables)
    )
    discriminatorOptimizer.apply_gradients(
        zip(gradientsOfDiscriminator, discriminator.trainable_variables)
    )


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for imageBatch in dataset:
            trainStep(imageBatch)

        # Save the model every epoch
        save_dir = (
            "%s/ML_ten_year_rainfall/fake/random_plain/dcgan/" + "Epoch_%04d"
        ) % (os.getenv("SCRATCH"), epoch,)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        checkpoint.save(file_prefix="%s/ckpt" % save_dir)

        print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))


# Train the GAN
train(trainingData, nEpochs)
