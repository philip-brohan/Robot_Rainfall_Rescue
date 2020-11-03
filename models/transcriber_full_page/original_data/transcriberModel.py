# Model specification based on the tuned ATB2 transcriber

import tensorflow as tf

# This file provides a subclass of tf.keras.Model that serves as a
#  transcriber for the rainfall images. It learns to make a tensor of
#  extracted digits from a tensor of the document image.

# import this file, instantiate an instance of the transcriberModel
#  class, and then either train it and save the weights, or load
#  pre-trained weights and use it for transcription.

# Model the image with hierachical convolutions and then map to output digits
class transcriberModel(tf.keras.Model):
    def __init__(self):
        # parent constructor
        super(transcriberModel, self).__init__()
        # Initial shape (1024,640,3)
        self.conv1A = tf.keras.layers.Conv2D(
            10, (3, 3), strides=(2, 2), padding="valid"
        )
        self.act1A = tf.keras.layers.ELU()
        # Now (512,320,10)
        self.conv1B = tf.keras.layers.Conv2D(
            10, (3, 3), strides=(2, 2), padding="valid"
        )
        self.act1B = tf.keras.layers.ELU()
        # Now (256,160,10)
        self.conv1C = tf.keras.layers.Conv2D(
            20, (3, 3), strides=(2, 2), padding="valid"
        )
        self.act1C = tf.keras.layers.ELU()
        # Now (128,40,20)
        self.conv1D = tf.keras.layers.Conv2D(
            40, (3, 3), strides=(2, 2), padding="valid"
        )
        self.act1D = tf.keras.layers.ELU()
        # Now (64,40,40)
        self.conv1E = tf.keras.layers.Conv2D(
            80, (3, 3), strides=(2, 2), padding="valid"
        )
        self.act1E = tf.keras.layers.ELU()
        # Now (32,20,80)
        # reshape to 1d
        self.flatten = tf.keras.layers.Flatten()
        # map directly to output format (520 digits)
        self.map_to_op = tf.keras.layers.Dense(
            520 * 11, kernel_regularizer=tf.keras.regularizers.l1(0.000001)
        )
        # softmax to get digit probabilities at each location
        self.op_reshape = tf.keras.layers.Reshape(target_shape=(520, 11,))
        self.op_softmax = tf.keras.layers.Softmax(axis=2)

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
        x = self.flatten(x)
        x = self.map_to_op(x)
        x = self.op_reshape(x)
        x = self.op_softmax(x)
        return x
