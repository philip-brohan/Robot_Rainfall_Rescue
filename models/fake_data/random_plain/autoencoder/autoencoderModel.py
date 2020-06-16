# Model specification for the logbook image autoencoder

import tensorflow as tf
import tensorflow.keras.backend as K

# This file provides a subclass of tf.keras.Model that serves as an
#  autoencoder for the ten-year rainfall images. The model has two sub-models
#  (an encoder and a generator) and is variational (adds noise to its
#  latent space) in training but not in prediction.

# import this file, instantiate an instance of the autoencoderModel
#  class, and then either train it and save the weights, or load
#  pre-trained weights and use it for prediction.

# Dimensionality of latent space
latent_dim = 512

# Specify and encoder model to pack the input into a latent space
class encoderModel(tf.keras.Model):
    def __init__(self):
        # parent constructor
        super(encoderModel, self).__init__()
        # Initial shape (1024,768,3)
        self.conv1A = tf.keras.layers.Conv2D(
            16, (3, 3), strides=(2, 2), padding="valid"
        )
        self.act1A = tf.keras.layers.ELU()
        # Now (512,384,16)
        self.conv1B = tf.keras.layers.Conv2D(
            32, (3, 3), strides=(2, 2), padding="valid"
        )
        self.act1B = tf.keras.layers.ELU()
        # Now (256,192,32)
        self.conv1C = tf.keras.layers.Conv2D(
            64, (3, 3), strides=(2, 2), padding="valid"
        )
        self.act1C = tf.keras.layers.ELU()
        # Now (128,96,64)
        self.conv1D = tf.keras.layers.Conv2D(
            128, (3, 3), strides=(2, 2), padding="valid"
        )
        self.act1D = tf.keras.layers.ELU()
        # Now (64,48,128)
        self.conv1E = tf.keras.layers.Conv2D(
            256, (3, 3), strides=(2, 2), padding="valid"
        )
        self.act1E = tf.keras.layers.ELU()
        # Now (32,24,256)
        self.conv1F = tf.keras.layers.Conv2D(
            512, (3, 3), strides=(2, 2), padding="valid"
        )
        self.act1F = tf.keras.layers.ELU()
        # Now (16,12,512)
        # reshape to 1d
        self.flatten = tf.keras.layers.Flatten()
        # reduce to latent space size
        self.pack_to_l = tf.keras.layers.Dense(latent_dim,)

    def call(self, inputs):
        x = self.conv1A(inputs)
        x = self.act1A(x)
        x = self.conv1B(x)
        x = self.act1B(x)
        x = self.conv1C(x)
        x = self.act1C(x)
        x = self.conv1D(x)
        x = self.act1D(x)
        x = self.conv1E(x)
        x = self.act1E(x)
        x = self.conv1F(x)
        x = self.act1F(x)
        x = self.flatten(x)
        x = self.pack_to_l(x)
        # Normalise latent space to mean=0, sd=1
        x = x - K.mean(x)
        x = x / K.std(x)
        return x


# Specify a generator model to make the output from a latent vector
class generatorModel(tf.keras.Model):
    def __init__(self):
        # parent constructor
        super(generatorModel, self).__init__()
        # reshape latent space as 3d seed for deconvolution
        self.unpack_from_l = tf.keras.layers.Dense(16 * 12 * 512,)
        self.unflatten = tf.keras.layers.Reshape(target_shape=(16, 12, 512,))
        # Starts at (16,12,512)
        self.conv1A = tf.keras.layers.Conv2DTranspose(
            256, (3, 3), strides=(2, 2), padding="same"
        )
        self.act1A = tf.keras.layers.ELU()
        # Now (32,24,256)
        self.conv1B = tf.keras.layers.Conv2DTranspose(
            128, (3, 3), strides=(2, 2), padding="same"
        )
        self.act1B = tf.keras.layers.ELU()
        # Now (64,48,128)
        self.conv1C = tf.keras.layers.Conv2DTranspose(
            64, (3, 3), strides=(2, 2), padding="same"
        )
        self.act1C = tf.keras.layers.ELU()
        # Now (128,96,64)
        self.conv1D = tf.keras.layers.Conv2DTranspose(
            32, (3, 3), strides=(2, 2), padding="same"
        )
        self.act1D = tf.keras.layers.ELU()
        # Now (256,192,32)
        self.conv1E = tf.keras.layers.Conv2DTranspose(
            16, (3, 3), strides=(2, 2), padding="same"
        )
        self.act1E = tf.keras.layers.ELU()
        # Now (512,384,16)
        self.conv1F = tf.keras.layers.Conv2DTranspose(
            3, (3, 3), strides=(2, 2), padding="same"
        )
        # Now back to (1024,768,3)

    def call(self, inputs):
        x = self.unpack_from_l(inputs)
        x = self.unflatten(x)
        x = self.conv1A(x)
        x = self.act1A(x)
        x = self.conv1B(x)
        x = self.act1B(x)
        x = self.conv1C(x)
        x = self.act1C(x)
        x = self.conv1D(x)
        x = self.act1D(x)
        x = self.conv1E(x)
        x = self.act1E(x)
        x = self.conv1F(x)
        return x


# Autoencoder model is the encoder and generator run in sequence
#  with some noise added between them in training.
class autoencoderModel(tf.keras.Model):
    def __init__(self):
        super(autoencoderModel, self).__init__()
        self.encoder = encoderModel()
        self.generator = generatorModel()
        self.noiseMean = tf.Variable(0.0, trainable=False)
        self.noiseStdDev = tf.Variable(0.5, trainable=False)

    def call(self, inputs, training=None):
        # encode to latent space
        x = self.encoder(inputs)
        #        if training:
        #            # Add noise to the latent space representation
        #            x += K.random_normal(
        #                K.shape(x),
        #                mean=K.get_value(self.noiseMean),
        #                stddev=K.get_value(self.noiseStdDev),
        #            )
        # Re-normalise latent space
        #            x = x - K.mean(x)
        #            x = x / K.std(x)
        # Generate real space representation from latent space
        x = self.generator(x)
        return x
