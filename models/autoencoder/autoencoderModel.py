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
latent_dim = 100

# Specify and encoder model to pack the input into a latent space
class encoderModel(tf.keras.Model):
    def __init__(self):
        # parent constructor
        super(encoderModel, self).__init__()
        # Initial shape (1024,640,1)
        self.conv1A = tf.keras.layers.Conv2D(
            16, (3, 3), strides=(2, 2), padding="same",
            activation=tf.keras.activations.elu
        )
        self.bn1A = tf.keras.layers.BatchNormalization()
        self.d1A = tf.keras.layers.SpatialDropout2D(0.5)
        # Now (512,320,16)
        self.conv1B = tf.keras.layers.Conv2D(
            32, (3, 3), strides=(2, 2), padding="same",
            activation=tf.keras.activations.elu
        )
        # Now (256,160,32)
        self.bn1B = tf.keras.layers.BatchNormalization()
        self.d1B = tf.keras.layers.SpatialDropout2D(0.5)
        self.conv1C = tf.keras.layers.Conv2D(
            64, (3, 3), strides=(2, 2), padding="same",
            activation=tf.keras.activations.elu
        )
        # Now (128,80,64)
        self.bn1C = tf.keras.layers.BatchNormalization()
        self.d1C = tf.keras.layers.SpatialDropout2D(0.5)
        self.conv1D = tf.keras.layers.Conv2D(
            128, (3, 3), strides=(2, 2), padding="same",
            activation=tf.keras.activations.elu
        )
        # Now (64,40,128)
        self.bn1D = tf.keras.layers.BatchNormalization()
        self.d1D = tf.keras.layers.SpatialDropout2D(0.5)
        # reshape to 1d
        self.flatten = tf.keras.layers.Flatten()
        # reduce to latent space size
        self.dense1 = tf.keras.layers.Dense(latent_dim,
            activation=tf.keras.activations.elu)
        # Normalise
        self.bnf = tf.keras.layers.BatchNormalization()


    def call(self, inputs):
        x = self.conv1A(inputs)
        x = self.bn1A(x)
        x = self.d1A(x)
        x = self.conv1B(x)
        x = self.bn1B(x)
        x = self.d1B(x)
        x = self.conv1C(x)
        x = self.bn1C(x)
        x = self.d1C(x)
        x = self.conv1D(x)
        x = self.bn1D(x)
        x = self.d1D(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bnf(x)
        return x


# Specify a generator model to make the output from a latent vector
class generatorModel(tf.keras.Model):
    def __init__(self):
        # parent constructor
        super(generatorModel, self).__init__()
        # Deconvolution starts at (64,40,512)
        # reshape latent space as 3d seed for deconvolution
        self.unpack_from_l = tf.keras.layers.Dense(64 * 40 * 128,
            activation=tf.keras.activations.elu)
        # reshape to 1d
        self.unflatten = tf.keras.layers.Reshape(target_shape=(64, 40, 128,))
        self.bn1U = tf.keras.layers.BatchNormalization()
        self.d1U = tf.keras.layers.SpatialDropout2D(0.5)
        self.conv1D = tf.keras.layers.Conv2DTranspose(
            64, (3, 3), strides=(2, 2), padding="same",
            activation=tf.keras.activations.elu
        )
        self.bn1D = tf.keras.layers.BatchNormalization()
        self.d1D = tf.keras.layers.SpatialDropout2D(0.5)
        # Now (128,80,64)
        self.conv1C = tf.keras.layers.Conv2DTranspose(
            32, (3, 3), strides=(2, 2), padding="same",
            activation=tf.keras.activations.elu
        )
        self.bn1C = tf.keras.layers.BatchNormalization()
        self.d1C = tf.keras.layers.SpatialDropout2D(0.5)
        # Now (256,160,32)
        self.conv1B = tf.keras.layers.Conv2DTranspose(
            16, (3, 3), strides=(2, 2), padding="same",
            activation=tf.keras.activations.elu
        )
        self.bn1B = tf.keras.layers.BatchNormalization()
        self.d1B = tf.keras.layers.SpatialDropout2D(0.5)
        # Now (512,320,16)
        self.conv1A = tf.keras.layers.Conv2DTranspose(
            1, (3, 3), strides=(2, 2), padding="same",
            activation=tf.keras.activations.elu
        )
        # Now back to (1024,640,1)

    def call(self, inputs):
        x = self.unpack_from_l(inputs)
        x = self.unflatten(x)
        x = self.bn1U(x)
        x = self.d1U(x)
        x = self.conv1D(x)
        x = self.bn1D(x)
        x = self.d1D(x)
        x = self.conv1C(x)
        x = self.bn1C(x)
        x = self.d1C(x)
        x = self.conv1B(x)
        x = self.bn1B(x)
        x = self.d1B(x)
        x = self.conv1A(x)
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
