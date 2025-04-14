# Model specification for a shallow convolutional image analyser

import tensorflow as tf

# Build the convolutional model
def dcCNR():

    inputs = tf.keras.Input(shape=(102, 64, 1))
    x=tf.keras.layers.GaussianNoise(0.001)(inputs)

    x = tf.keras.layers.Conv2D(
        16,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
    )(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.ELU()(x)

    x = tf.keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
    )(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.ELU()(x)

    # Map to desird outputs
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(8)(x)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=x, name="dcCNR")
    return model
