from keras import Model, layers


def make_allcnn():
    inputs = layers.Input((32, 32, 3))

    out_channels = [96, 96, 96, 192, 192, 192, 192, 192, 10]
    ksizes = [3, 3, 3, 3, 3, 3, 3, 1, 1]
    strides = [1, 1, 2, 1, 1, 2, 1, 1, 1]

    x = layers.Dropout(0.2)(inputs)
    for out_c, ksize, stride in zip(out_channels, ksizes, strides):
        padding = "valid" if (ksize == 1) else "same"
        x = layers.Conv2D(out_c, ksize, stride, padding=padding)(x)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)

        if stride == 2:
            x = layers.Dropout(0.5)(x)

    x = layers.GlobalAveragePooling2D()(x)

    allcnn = Model(inputs=inputs, outputs=x)
    return allcnn
