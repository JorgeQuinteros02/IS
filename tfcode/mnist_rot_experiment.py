import time
import tensorflow as tf
import keras
from splitconv2d import P4ConvP4, P4ConvZ2
import csv


def make_z2cnn():
    z2cnn_model = keras.Sequential()

    z2cnn_model.add(keras.Input(shape=(28,28, 1)))

    for i in range(6):
        z2cnn_model.add(keras.layers.Conv2D(20,
                                            kernel_size=3,
                                            strides=1,
                                            activation="relu",
                                            name="conv" + str(i + 1)))
        z2cnn_model.add(keras.layers.Dropout(rate=0.3))
        if i == 1:
            z2cnn_model.add(keras.layers.MaxPooling2D())
    z2cnn_model.add(keras.layers.Conv2D(10,
                                        kernel_size=4,
                                        name="conv7"))

    z2cnn_model.add(keras.layers.Flatten())
    z2cnn_model.add(keras.layers.Softmax())

    return z2cnn_model

def make_p4cnn():

    p4cnn_model = keras.Sequential()

    p4cnn_model.add(keras.Input(shape=(28,28, 1)))

    p4cnn_model.add(P4ConvZ2(10,
                            kernel_size=3,
                            stride=1,
                            name="conv1"))
    p4cnn_model.add(keras.layers.Activation('relu'))
    p4cnn_model.add(keras.layers.Dropout(rate=0.3))
    for i in range(2, 7):
        p4cnn_model.add(P4ConvP4(10,
                                kernel_size=3,
                                stride=1,
                                name="conv" + str(i)))
        
        p4cnn_model.add(keras.layers.Activation('relu'))
        p4cnn_model.add(keras.layers.Dropout(rate=0.3))
        if i == 2:
            p4cnn_model.add(keras.layers.MaxPooling2D())
    p4cnn_model.add(P4ConvP4(10,
                                        kernel_size=4,
                                        name="conv7"))

    p4cnn_model.add(keras.layers.GlobalMaxPool3D(keepdims=False))

    return p4cnn_model

if __name__ == "__main__":
    (x_test, y_test), (x_train, y_train) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(10000, 28, 28, 1).astype("float32") / 255
    x_test = x_test.reshape(60000, 28, 28, 1).astype("float32") / 255

    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    x_val = x_test[:2000]
    y_val = y_test[:2000]
    x_test = x_test[2000:]
    y_test = y_test[2000:]

    data_augmentation = keras.Sequential([
        keras.layers.RandomRotation(0.5)
    ])

    x_train = data_augmentation(x_train)
    x_test = data_augmentation(x_test)
    x_val = data_augmentation(x_val)



    models = {"z2c": make_z2cnn,
              "p4": make_p4cnn}
    epochs = 100
    repeats = 10

    starttime = time.time()



    for i in range(repeats):
        tf.random.set_seed(i)
        for key in models:
            model = models[key]()
            model.compile(
                optimizer=keras.optimizers.Adam(),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[keras.metrics.sparse_categorical_accuracy]
            )
            model.fit(
                x_train,
                y_train,
                epochs=epochs,
                validation_data=(x_val, y_val)
            )
            loss, accuracy = model.evaluate(x_test, y_test)

            with open("mnist_rot" + str(starttime) + ".txt", "a") as file:
                f = csv.writer(file)
                f.writerow([key, i, epochs, loss, accuracy])

                

            del model

    endtime = time.time()