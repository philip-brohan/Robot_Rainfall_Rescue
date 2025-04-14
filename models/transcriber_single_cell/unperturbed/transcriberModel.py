# Model specification for a single cell image transcriber

import tensorflow as tf

# This file provides a subclass of tf.keras.Model that serves as a
#  transcriber for the ATB2 images. It learns to make a tensor of
#  extracted digits from a tensor of the document image.

# import this file, instantiate an instance of the transcriberModel
#  class, and then either train it and save the weights, or load
#  pre-trained weights and use it for transcription.

# Model the image with hierachical convolutions and then map to output digits
class transcriberModel(tf.keras.Model):
    def __init__(self):
        # parent constructor
        super(transcriberModel, self).__init__()
        # Initial shape (64,64,1)
        self.conv1A = tf.keras.layers.Conv2D(
            10, (3, 3), strides=(2, 2), padding="valid"
        )
        self.act1A = tf.keras.layers.ELU()
        # Now (32,32,10)
        self.conv1B = tf.keras.layers.Conv2D(
            20, (3, 3), strides=(2, 2), padding="valid"
        )
        self.act1B = tf.keras.layers.ELU()
        # Now (16,16,20)
        # reshape to 1d
        self.flatten = tf.keras.layers.Flatten()
        # map directly to output format (436 digits)
        self.map_to_op = tf.keras.layers.Dense(
            3 * 10, kernel_regularizer=tf.keras.regularizers.l1(0.000001)
        )
        # softmax to get digit probabilities at each location
        self.op_reshape = tf.keras.layers.Reshape(
            target_shape=(
                3,
                10,
            )
        )
        self.op_softmax = tf.keras.layers.Softmax(axis=2)

    def call(self, inputs):
        x = self.conv1A(inputs)
        x = self.act1A(x)
        x = self.conv1B(x)
        x = self.act1B(x)
        x = self.flatten(x)
        x = self.map_to_op(x)
        x = self.op_reshape(x)
        x = self.op_softmax(x)
        return x
