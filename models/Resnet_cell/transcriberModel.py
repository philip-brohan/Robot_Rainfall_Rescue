# Model specification for a resnet-based image transcriber
# Based on https://towardsdatascience.com/understand-and-implement-resnet-50-with-tensorflow-2-0-1190b9b52691

import tensorflow as tf

# Define two sorts of residual block
# One which keeps the tensor dimension fixed (Identity)
#  and one shrinks the dimension by nxn (Stride)


def resIdentity(x, filters, reg=tf.keras.regularizers.l2(0.00), dropout=0.0):

    # x is the input tensor
    x_skip = x  # Keep this for combination with output (what makes it residual).

    # filters is a 2-element array giving numbers of features
    # nFProc - no. of features used in block processing
    # nFOut - no. of features on input and output
    nFProc, nFIO = filters

    # Map input to nFProc features with 1x1 convolution
    x = tf.keras.layers.Conv2D(
        nFProc,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        kernel_regularizer=reg,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.SpatialDropout2D(dropout)(x)

    # Apply a 3x3 convolution (but size kept same with padding)
    x = tf.keras.layers.Conv2D(
        nFProc,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_regularizer=reg,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.SpatialDropout2D(dropout)(x)

    # Resize back to original no. of features
    x = tf.keras.layers.Conv2D(
        nFIO,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        kernel_regularizer=reg,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Add the input back (residual)
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.ReLU()(x)

    return x


def resStride(x, s, filters, reg=tf.keras.regularizers.l2(0.00), dropout=0.0):

    # Exactly the same as resIdentity, except that both original and residual are
    #  reduced by s*s, using strided 1xi convolution
    x_skip = x
    nFProc, nFIO = filters

    # Map input to nFProc features with 1x1 convolution
    x = tf.keras.layers.Conv2D(
        nFProc,
        kernel_size=(1, 1),
        strides=(s, s),
        padding="valid",
        kernel_regularizer=reg,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.SpatialDropout2D(dropout)(x)

    # Apply a 3x3 convolution (but size kept same with padding)
    x = tf.keras.layers.Conv2D(
        nFProc,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_regularizer=reg,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # x = tf.keras.layers.SpatialDropout2D(dropout)(x)

    # Resize back to original no. of features
    x = tf.keras.layers.Conv2D(
        nFIO,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        kernel_regularizer=reg,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Add the input back (residual)
    x_skip = tf.keras.layers.Conv2D(
        nFIO,
        kernel_size=(1, 1),
        strides=(s, s),
        padding="valid",
        kernel_regularizer=reg,
    )(x_skip)
    x_skip = tf.keras.layers.BatchNormalization()(x_skip)
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.ReLU()(x)

    return x


# Build the residual model
def resnetRRR():

    # Reduce the dimensions of the input with a large-scale strided convolution
    #  and maxpooling - modelled after resnet50
    inputs = tf.keras.Input(shape=(64, 64, 1))
    x = tf.keras.layers.Conv2D(
        16,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding="valid",
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Now build a deep residual model on the reduced inputs
    x = resStride(x, s=1, filters=(16, 32))
    x = resIdentity(x, filters=(16, 32))
    x = resIdentity(x, filters=(16, 32))

    x = resStride(x, s=2, filters=(32, 64))
    x = resIdentity(x, filters=(32, 64))
    x = resIdentity(x, filters=(32, 64))
    x = resIdentity(x, filters=(32, 64))

    #    x = resStride(x, s=2, filters=(64, 128))
    #    x = resIdentity(x, filters=(64, 128))
    #    x = resIdentity(x, filters=(64, 128))
    #    x = resIdentity(x, filters=(64, 128))

    #    x = resStride(x, s=2, filters=(128, 256))
    #    x = resIdentity(x, filters=(128, 256))
    #    x = resIdentity(x, filters=(128, 256))
    #    x = resIdentity(x, filters=(128, 256))

    # Map input to 30 features with 1x1 convolution
    x = tf.keras.layers.Conv2D(
        30,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # Reduce to a vector with global average pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Convert to digit probabilities with softmax
    x = tf.keras.layers.Reshape(target_shape=(3, 10))(x)
    x = tf.keras.layers.Softmax(axis=2)(x)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=x, name="ResnetRRR")
    return model
