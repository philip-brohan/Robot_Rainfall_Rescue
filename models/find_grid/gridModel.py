# Model specification for the ATB2 image grid-corner seeker

import tensorflow as tf

# This file provides a subclass of tf.keras.Model that serves as a
#  analyser for the ATB2 images. It learns to make a tensor of
#  grid-box centre locations from a tensor of the document image.

# import this file, instantiate an instance of the gridModel
#  class, and then either train it and save the weights, or load
#  pre-trained weights and use it for analysis.

# Model the image with hierachical convolutions and then map to corner coordinates
class gridModel(tf.keras.Model):
    def __init__(self):
        # parent constructor
        super(gridModel, self).__init__()
        # Initial shape (1024,640,1)
        self.conv1A = tf.keras.layers.Conv2D(
            64, (3, 3), strides=(2, 2), padding="valid"
        )
        self.drop1A = tf.keras.layers.Dropout(0.3)
        self.act1A = tf.keras.layers.ELU()
        # Now (512,384,10)
        self.conv1B = tf.keras.layers.Conv2D(
            32, (3, 3), strides=(2, 2), padding="valid"
        )
        self.act1B = tf.keras.layers.ELU()
        self.drop1B = tf.keras.layers.Dropout(0.3)
        self.conv1C = tf.keras.layers.Conv2D(
            16, (3, 3), strides=(2, 2), padding="valid"
        )
        self.drop1C = tf.keras.layers.Dropout(0.3)
        self.act1C = tf.keras.layers.ELU()
        # Now (128,96,64)
        self.conv1D = tf.keras.layers.Conv2D(8, (3, 3), strides=(2, 2), padding="valid")
        self.drop1D = tf.keras.layers.Dropout(0.3)
        self.act1D = tf.keras.layers.ELU()
        # Now (64,48,128)
        self.conv1E = tf.keras.layers.Conv2D(4, (3, 3), strides=(2, 2), padding="valid")
        self.drop1E = tf.keras.layers.Dropout(0.3)
        self.act1E = tf.keras.layers.ELU()
        # Now (32,24,256)
        # self.conv1F = tf.keras.layers.Conv2D(
        #    512, (3, 3), strides=(2, 2), padding="valid"
        # )
        # self.drop1F = tf.keras.layers.Dropout(0.3)
        # self.act1F = tf.keras.layers.ELU()
        # Now (16,12,512)
        # reshape to 1d
        self.flatten = tf.keras.layers.Flatten()
        # 2-layer output assesment
        self.map1 = tf.keras.layers.Dense(
            240,
        )
        self.actm1 = tf.keras.layers.ELU()
        self.actm1d = tf.keras.layers.Dropout(0.5)
        # map directly to output format (240 coordinates)
        self.map_to_op = tf.keras.layers.Dense(
            240,
        )
        # Want float32 output even if using lower precision
        self.opl = tf.keras.layers.Activation("linear", dtype="float32")

    def call(self, inputs):
        x = self.conv1A(inputs)
        x = self.drop1A(x)
        x = self.act1A(x)
        x = self.conv1B(x)
        x = self.drop1B(x)
        x = self.act1B(x)
        #x = self.conv1C(x)
        #x = self.drop1C(x)
        #x = self.act1C(x)
        # x = self.conv1D(x)
        # x = self.drop1D(x)
        # x = self.act1D(x)
        # x = self.conv1E(x)
        # x = self.drop1E(x)
        # x = self.act1E(x)
        # x = self.conv1F(x)
        # x = self.drop1F(x)
        # x = self.act1F(x)
        x = self.flatten(x)
        x = self.map1(x)
        x = self.actm1(x)
        x = self.actm1d(x)
        x = self.map_to_op(x)
        x = self.opl(x)
        return x
