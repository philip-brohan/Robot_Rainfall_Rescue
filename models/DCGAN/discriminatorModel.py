# Model specification for the ten-year rainfall image discriminator

import tensorflow as tf

# This file provides a subclass of tf.keras.Model that serves as an
#  discriminator for the ten-year rainfall images.

# import this file, instantiate an instance of the discriminatorModel
#  class, and then train it alongside the generator.

# Specify and encoder model to pack the input into a latent space
class discriminatorModel(tf.keras.Model):
    def __init__(self):
        # parent constructor
        super(discriminatorModel, self).__init__()
        # Initial shape (1024,640,1)
        self.conv1A = tf.keras.layers.Conv2D(
            10,
            (3, 3),
            strides=(2, 2),
            padding="same",
            activation=tf.keras.activations.elu,
        )
        self.drop1A = tf.keras.layers.Dropout(0.3)
        # Now (512,320,80)
        self.conv1B = tf.keras.layers.Conv2D(
            10,
            (3, 3),
            strides=(2, 2),
            padding="same",
            activation=tf.keras.activations.elu,
        )
        self.drop1B = tf.keras.layers.Dropout(0.3)
        # Now (256,160,40)
        self.conv1C = tf.keras.layers.Conv2D(
            10,
            (3, 3),
            strides=(2, 2),
            padding="same",
            activation=tf.keras.activations.elu,
        )
        self.drop1C = tf.keras.layers.Dropout(0.3)
        # Now (128,80,20)
        self.conv1D = tf.keras.layers.Conv2D(
            10,
            (3, 3),
            strides=(2, 2),
            padding="same",
            activation=tf.keras.activations.elu,
        )
        self.drop1D = tf.keras.layers.Dropout(0.3)
        # Now (10,40,128)
        # reshape to 1d
        self.flatten = tf.keras.layers.Flatten()
        # Single output - true/false classifier
        self.opl = tf.keras.layers.Dense(1, activation=tf.keras.activations.elu)

    def call(self, inputs):
        x = self.conv1A(inputs)
        x = self.drop1A(x)
        x = self.conv1B(x)
        x = self.drop1B(x)
        x = self.conv1C(x)
        x = self.drop1C(x)
        x = self.conv1D(x)
        x = self.drop1D(x)
        x = self.flatten(x)
        x = self.opl(x)
        return x


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Discriminator needs to judge real images as real and generated images as fake
# realOutput is the discriminator's opinion on a set of real images
# fakeOutput is the discriminator's opinion on a set of generated images
def discriminatorLoss(realOutput, fakeOutput):
    real_loss = cross_entropy(tf.ones_like(realOutput), realOutput)
    fake_loss = cross_entropy(tf.zeros_like(fakeOutput), fakeOutput)
    total_loss = real_loss + fake_loss
    return total_loss


# Optimiser
discriminatorOptimizer = tf.keras.optimizers.Adam(1e-4)
