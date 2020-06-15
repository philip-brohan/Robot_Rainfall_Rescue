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
        # Initial shape (1024,768,3)
        self.conv1A = tf.keras.layers.Conv2D(
            16, (3, 3), strides=(2, 2), padding="valid"
        )
        self.drop1A = tf.keras.layers.Dropout(0.3)
        self.act1A = tf.keras.layers.ELU()
        # Now (512,384,16)
        self.conv1B = tf.keras.layers.Conv2D(
            32, (3, 3), strides=(2, 2), padding="valid"
        )
        self.drop1B = tf.keras.layers.Dropout(0.3)
        self.act1B = tf.keras.layers.ELU()
        # Now (256,192,32)
        self.conv1C = tf.keras.layers.Conv2D(
            64, (3, 3), strides=(2, 2), padding="valid"
        )
        self.drop1C = tf.keras.layers.Dropout(0.3)
        self.act1C = tf.keras.layers.ELU()
        # Now (128,96,64)
        self.conv1D = tf.keras.layers.Conv2D(
            128, (3, 3), strides=(2, 2), padding="valid"
        )
        self.drop1D = tf.keras.layers.Dropout(0.3)
        self.act1D = tf.keras.layers.ELU()
        # Now (64,48,128)
        self.conv1E = tf.keras.layers.Conv2D(
            256, (3, 3), strides=(2, 2), padding="valid"
        )
        self.drop1E = tf.keras.layers.Dropout(0.3)
        self.act1E = tf.keras.layers.ELU()
        # Now (32,24,256)
        self.conv1F = tf.keras.layers.Conv2D(
            512, (3, 3), strides=(2, 2), padding="valid"
        )
        self.drop1F = tf.keras.layers.Dropout(0.3)
        self.act1F = tf.keras.layers.ELU()
        # Now (16,12,512)
        # reshape to 1d
        self.flatten = tf.keras.layers.Flatten()
        # Single output - true/false classifier
        self.opl = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.conv1A(inputs)
        x = self.drop1A(x)
        x = self.act1A(x)
        x = self.conv1B(x)
        x = self.drop1B(x)
        x = self.act1B(x)
        x = self.conv1C(x)
        x = self.drop1C(x)
        x = self.act1C(x)
        x = self.conv1D(x)
        x = self.drop1D(x)
        x = self.act1D(x)
        x = self.conv1E(x)
        x = self.drop1E(x)
        x = self.act1E(x)
        x = self.conv1F(x)
        x = self.drop1F(x)
        x = self.act1F(x)
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
