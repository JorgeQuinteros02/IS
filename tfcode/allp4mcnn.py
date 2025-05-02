from keras import layers, Model
from keras_gcnn.layers import GConv2D


# return ALL-CNN with all convolutions replaced by p4m-convolutions
def make_allp4mcnn():
    inputs = layers.Input((32, 32, 3))

    out_channels = [32, 32, 32, 64, 64, 64, 64, 64, 10]
    ksizes = [3, 3, 3, 3, 3, 3, 3, 1, 1]
    strides = [1, 1, 2, 1, 1, 2, 1, 1, 1]

    x = layers.Dropout(0.2)(inputs)
    h_in = 'Z2'
    for out_c, ksize, stride in zip(out_channels, ksizes, strides):
        padding = "valid" if (ksize == 1) else "same"
        x = GConv2D(out_c, ksize, h_in, 'D4', stride, padding=padding)(x)
        h_in = 'D4'
        x = layers.Activation('relu')(x)

        if stride == 2:
            x = layers.Dropout(0.5)(x)

    x = layers.Reshape((8, 8, 8, 10))(x)
    x = layers.GlobalAveragePooling3D()(x)

    allp4mcnn = Model(inputs=inputs, outputs=x)
    return allp4mcnn
