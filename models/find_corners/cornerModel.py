# Model specification for the ATB2 image grid-corner seeker

import tensorflow as tf

# This file provides a subclass of tf.keras.Model that serves as a
#  analyser for the ATB2 images. It learns to make a tensor of
#  grid corner locations from a tensor of the document image.

# import this file, instantiate an instance of the cornerModel
#  class, and then either train it and save the weights, or load
#  pre-trained weights and use it for analyis.

# Model the image with hierachical convolutions and then map to corner coordinates
class cornerModel(tf.keras.Model):
    def __init__(self):
        # parent constructor
        super(cornerModel, self).__init__()
        # Initial shape (1024,768,1)
        self.conv1A = tf.keras.layers.Conv2D(
            20, (3, 3), strides=(2, 2), padding="valid"
        )
        self.drop1A = tf.keras.layers.Dropout(0.5)
        self.act1A = tf.keras.layers.ELU()
        # Now (512,384,10)
        self.conv1B = tf.keras.layers.Conv2D(
            40, (3, 3), strides=(2, 2), padding="valid"
        )
        self.act1B = tf.keras.layers.ELU()
        self.drop1b = tf.keras.layers.Dropout(0.5)
        # reshape to 1d
        self.flatten = tf.keras.layers.Flatten()
        # map directly to output format (22 coordinates)
        self.map_to_op = tf.keras.layers.Dense(
            44,
        )

    def call(self, inputs):
        x = self.conv1A(inputs)
        x = self.drop1A(x)
        x = self.act1A(x)
        x = self.conv1B(x)
        x = self.drop1A(x)
        x = self.act1B(x)
        x = self.flatten(x)
        x = self.map_to_op(x)
        return x
