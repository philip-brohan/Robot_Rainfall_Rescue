# Model specification for a resnet-based image transcriber
# Based on https://towardsdatascience.com/understand-and-implement-resnet-50-with-tensorflow-2-0-1190b9b52691

import tensorflow as tf

# Define two sorts of residual block
# One which keeps the tensor dimension fixed (Identity)
#  and one shrinks the dimension by nxn (Stride)


def resIdentity(x, filters, reg=tf.keras.regularizers.l2(0.001)):

    # x is the input tensor
    x_skip = x  # Keep this for combination with output (what makes it residual).

    # filters is a 2-element array giving numbers of features
    # nFProc - no. of features used in block processing
    # nFOut - no. of features on input and output
    nFProc, nFIO = filters

    # Map input to nFProc features with 1x1 convolution
    x = tf.keras.layers.Conv2D(
        nFIn,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        kernel_regularizer=reg,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

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

    # Resize back to original no. of features
    x = tf.keras.layers.Conv2D(
        nFOut,
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


def resStride(x, s, filters, reg=tf.keras.regularizers.l2(0.001)):

    # Exactly the same as resIdentity, except that both original and residual are
    #  reduced by s*s, using strided 1xi convolution

    # Map input to nFProc features with 1x1 convolution
    x = tf.keras.layers.Conv2D(
        nFIn,
        kernel_size=(1, 1),
        strides=(s, s),
        padding="valid",
        kernel_regularizer=reg,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

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

    # Resize back to original no. of features
    x = tf.keras.layers.Conv2D(
        nFOut,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        kernel_regularizer=reg,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Add the input back (residual)
    x_skip = tf.keras.layers.Conv2D(
        nFIn,
        kernel_size=(1, 1),
        strides=(s, s),
        padding="valid",
        kernel_regularizer=reg,
    )(x)
    x_skip = tf.keras.layers.BatchNormalization()(x_skip)
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.ReLU()(x)

    return x


# Build the residual model
def resnetRRR():

    # Reduce the dimensions of the input with a large-scale strided convolution
    #  and maxpooling - modelled after resnet50
    x = tf.keras.layers.Conv2D(16, kernel_size=(7, 7), strides=(2, 2), padding="valid")(
        tf.keras.Input()
    )
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Now build a deep residual model on the reduced inputs
    x = resStride(x, s=1, filters=(16, 32))
    x = resIdentity(x, filters=(16, 32))
    x = resIdentity(x, filters=(16, 32))

    x = resStride(x, s=2, filters=(32, 64))
    x = resIdentity(x, filters=(32, 64))
    x = resIdentity(x, filters=(32, 64))
    x = resIdentity(x, filters=(32, 64))

    x = resStride(x, s=2, filters=(64, 128))
    x = resIdentity(x, filters=(64, 128))
    x = resIdentity(x, filters=(64, 128))
    x = resIdentity(x, filters=(64, 128))

    x = resStride(x, s=2, filters=(128, 256))
    x = resIdentity(x, filters=(128, 256))
    x = resIdentity(x, filters=(128, 256))
    x = resIdentity(x, filters=(128, 256))

    # Map to output
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(436 * 10)(x)
    x = tf.keras.layers.Reshape(target_shape=(436, 10))(x)
    x = tf.keras.layers.Softmax(axis=2)(x)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=x, name="ResnetRRR")
    return model
