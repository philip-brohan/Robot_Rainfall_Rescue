# Model specification for the logbook image filter
#  Basically this is an autoencoder with a convolutional layer with few filters

import tensorflow as tf

# Specify and encoder model to pack the input into a latent space
class encoderModel(tf.keras.Model):
    def __init__(self):
        # parent constructor
        super(encoderModel, self).__init__()
        # Initial shape (1024,640,1)
        self.conv1A = tf.keras.layers.Conv2D(
            80, (3, 3), strides=(2, 2), padding="same",
            activation=tf.keras.activations.elu
        )
        self.conv1B = tf.keras.layers.Conv2D(
            40, (3, 3), strides=(2, 2), padding="same",
            activation=tf.keras.activations.elu
        )
        self.conv1C = tf.keras.layers.Conv2D(
            20, (3, 3), strides=(2, 2), padding="same",
            activation=tf.keras.activations.elu
        )
        self.conv1D = tf.keras.layers.Conv2D(
            10, (3, 3), strides=(2, 2), padding="same",
            activation=tf.keras.activations.elu
        )
        self.conv1E = tf.keras.layers.Conv2D(
            10, (3, 3), strides=(2, 2), padding="same",
            activation=tf.keras.activations.elu
        )
        self.conv1F = tf.keras.layers.Conv2D(
            10, (3, 3), strides=(2, 2), padding="same",
            activation=tf.keras.activations.elu
        )

    def call(self, inputs):
        x = self.conv1A(inputs)
        x = self.conv1B(x)
        x = self.conv1C(x)
        x = self.conv1D(x)
        #x = self.conv1E(x)
        #x = self.conv1F(x)
        return x


# Specify a generator model to make the output from a latent vector
class generatorModel(tf.keras.Model):
    def __init__(self):
        # parent constructor
        super(generatorModel, self).__init__()
        self.conv1A = tf.keras.layers.Conv2DTranspose(
            20, (3, 3), strides=(2, 2), padding="same",
            activation=tf.keras.activations.elu
        )
        self.conv1B = tf.keras.layers.Conv2DTranspose(
            40, (3, 3), strides=(2, 2), padding="same",
            activation=tf.keras.activations.elu
        )
        self.conv1C = tf.keras.layers.Conv2DTranspose(
            80, (3, 3), strides=(2, 2), padding="same",
            activation=tf.keras.activations.elu
        )
        self.conv1D = tf.keras.layers.Conv2DTranspose(
            10, (3, 3), strides=(2, 2), padding="same",
            activation=tf.keras.activations.elu
        )
        self.conv1E = tf.keras.layers.Conv2DTranspose(
            10, (3, 3), strides=(2, 2), padding="same",
            activation=tf.keras.activations.elu
        )
        self.conv1F = tf.keras.layers.Conv2DTranspose(
            1, (3, 3), strides=(2, 2), padding="same",
            activation=tf.keras.activations.elu
        )

    def call(self, inputs):
        x = self.conv1A(inputs)
        x = self.conv1B(x)
        x = self.conv1C(x)
        #x = self.conv1D(x)
        #x = self.conv1E(x)
        x = self.conv1F(x)
        return x


# Filter model is the encoder and generator run in sequence.
class filterModel(tf.keras.Model):
    def __init__(self):
        super(filterModel, self).__init__()
        self.encoder = encoderModel()
        self.generator = generatorModel()

    def call(self, inputs, training=None):
        x = self.encoder(inputs)
        x = self.generator(x)
        return x
