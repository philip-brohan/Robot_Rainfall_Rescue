# Model specification for the ten-year rainfall image generator

import tensorflow as tf

# This file provides a subclass of tf.keras.Model that serves as an
#  generator for the ten-year rainfall images.

# import this file, instantiate an instance of the generatorModel
#  class, and then either train it and save the weights, or load
#  pre-trained weights and use it for prediction.

# Specify a generator model to make the output from a latent vector
class generatorModel(tf.keras.Model):
    def __init__(self):
        # parent constructor
        super(generatorModel, self).__init__()
        # reshape latent space as 3d seed for deconvolution
        self.unflatten = tf.keras.layers.Reshape(target_shape=(16, 10, 512,))
        # Starts at (16,10,512)
        self.conv1A = tf.keras.layers.Conv2DTranspose(
            256,
            (3, 3),
            strides=(2, 2),
            padding="same",
            activation=tf.keras.activations.elu,
        )
        self.norm1A = tf.keras.layers.BatchNormalization()
        # Now (32,20,256)
        self.conv1B = tf.keras.layers.Conv2DTranspose(
            128,
            (3, 3),
            strides=(2, 2),
            padding="same",
            activation=tf.keras.activations.elu,
        )
        self.norm1B = tf.keras.layers.BatchNormalization()
        # Now (64,40,128)
        self.conv1C = tf.keras.layers.Conv2DTranspose(
            64,
            (3, 3),
            strides=(2, 2),
            padding="same",
            activation=tf.keras.activations.elu,
        )
        self.norm1C = tf.keras.layers.BatchNormalization()
        # Now (128,80,64)
        self.conv1D = tf.keras.layers.Conv2DTranspose(
            32,
            (3, 3),
            strides=(2, 2),
            padding="same",
            activation=tf.keras.activations.elu,
        )
        self.norm1D = tf.keras.layers.BatchNormalization()
        # Now (256,160,32)
        self.conv1E = tf.keras.layers.Conv2DTranspose(
            16,
            (3, 3),
            strides=(2, 2),
            padding="same",
            activation=tf.keras.activations.elu,
        )
        self.norm1E = tf.keras.layers.BatchNormalization()
        # Now (512,320,16)
        self.conv1F = tf.keras.layers.Conv2DTranspose(
            1,
            (3, 3),
            strides=(2, 2),
            padding="same",
            activation=tf.keras.activations.elu,
        )
        # Now (1024,640,1)

    def call(self, inputs):
        x = self.unflatten(x)
        x = self.conv1A(x)
        x = self.norm1A(x)
        x = self.conv1B(x)
        x = self.norm1B(x)
        x = self.conv1C(x)
        x = self.norm1C(x)
        x = self.conv1D(x)
        x = self.norm1D(x)
        x = self.conv1E(x)
        x = self.norm1E(x)
        x = self.conv1F(x)
        return x


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Fake output is the discriminator's opinion of the generated images
# This measures how successful we were in tricking the discriminator
def generatorLoss(fakeOutput):
    return cross_entropy(tf.ones_like(fakeOutput), fakeOutput)


# Optimizer
generatorOptimizer = tf.keras.optimizers.Adam(1e-4)
