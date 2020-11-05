#!/usr/bin/env python

# Deep Convolutional GAN for ten-year rainfall images.

import os
import sys
import time
import tensorflow as tf
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
from generatorModel import generatorModel, generatorLoss, generatorOptimizer
from discriminatorModel import (
    discriminatorModel,
    discriminatorLoss,
    discriminatorOptimizer,
)

# Load the data source provider
sys.path.append("%s/../../dataset" % os.path.dirname(__file__))
from makeRRDataset import getImageDataset

# How many epochs to train for
nEpochs = 500

# Latent space dimension
latentDim = 100

# How many images to use?
nTrainingImages = 11686  # Max is 11686

# Dataset parameters
bufferSize = nTrainingImages
batchSize = 32

# Set up the training data
trainingData = getImageDataset(purpose="training", nImages=nTrainingImages)
trainingData = trainingData.shuffle(bufferSize).batch(batchSize)

# Instantiate the models
with strategy.scope():
    generator = generatorModel()
    discriminator = discriminatorModel()

# Specify what to save at a checkpoint
checkpoint = tf.train.Checkpoint(
    generatorOptimizer=generatorOptimizer,
    discriminatorOptimizer=discriminatorOptimizer,
    generator=generator,
    discriminator=discriminator,
)
    
# If we are doing a restart, load from checkpoint
if args.epoch > 0:
    save_dir = ("%s/ML_ten_year_rainfall/models/DCGAN/original/Epoch_%04d") % (
        os.getenv("SCRATCH"),
        args.epoch-1,
    )
    status = checkpoint.restore(tf.train.latest_checkpoint(save_dir))

# Explicit training loop
@tf.function
def trainStep(images):
    noise = tf.random.normal([batchSize, 16*10*512])

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
        save_dir = ("%s/ML_ten_year_rainfall/models/DCGAN/original/Epoch_%04d") % (
            os.getenv("SCRATCH"),
            epoch,
        )
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        checkpoint.save(file_prefix="%s/ckpt" % save_dir)

        print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))


# Train the GAN
train(trainingData, nEpochs)
