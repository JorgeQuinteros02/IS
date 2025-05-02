from keras import Model, layers
from keras_gcnn.layers import GConv2D


def make_allp4cnn():
    inputs = layers.Input((32, 32, 3))

    out_channels = [48, 48, 48, 96, 96, 96, 96, 96, 10]
    ksizes = [3, 3, 3, 3, 3, 3, 3, 1, 1]
    strides = [1, 1, 2, 1, 1, 2, 1, 1, 1]

    x = layers.Dropout(0.2)(inputs)
    h_in = 'Z2'
    for out_c, ksize, stride in zip(out_channels, ksizes, strides):
        padding = "valid" if (ksize == 1) else "same"
        x = GConv2D(out_c, ksize, h_in, 'C4', stride, padding=padding)(x)
        h_in = 'C4'
        x = layers.Activation('relu')(x)

        if stride == 2:
            x = layers.Dropout(0.5)(x)

    x = layers.Reshape((8, 8, 4, 10))(x)
    x = layers.GlobalAveragePooling3D()(x)

    allp4cnn = Model(inputs=inputs, outputs=x)
    return allp4cnn