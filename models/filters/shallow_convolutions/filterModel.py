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
            10, (3, 3), strides=(2, 2), padding="valid"
        )
        # Now (512,320,10)
        self.act1A = tf.keras.layers.ELU()
        # reshape to 1d
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        x = self.conv1A(inputs)
        #x = self.act1A(x)
        x = self.flatten(x)
        return x


# Specify a generator model to make the output from a latent vector
class generatorModel(tf.keras.Model):
    def __init__(self):
        # parent constructor
        super(generatorModel, self).__init__()
        # reshape latent space as 3d seed for deconvolution
        self.unflatten = tf.keras.layers.Reshape(target_shape=(512, 320, 10,))
        # Now (512,320,10)
        self.conv1A = tf.keras.layers.Conv2DTranspose(
            1, (3, 3), strides=(2, 2), padding="same"
        )
        # Now back to (1024,640,1)

    def call(self, inputs):
        x = self.unflatten(inputs)
        x = self.conv1A(x)
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
