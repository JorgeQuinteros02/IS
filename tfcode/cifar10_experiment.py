import numpy as np
import tensorflow as tf
from keras import optimizers, losses, metrics, callbacks
from data.cifar.train import get_cifar10_data
from allcnn import make_allcnn
from allp4cnn import make_allp4cnn
from allp4mcnn import make_allp4mcnn


def run_cifar10():
    train_data, train_labels, val_data, val_labels = get_cifar10_data(
        datadir="./data/cifar",
        trainfn="train_all.npz",
        valfn="test.npz"
    )

    img = train_data[8000]

    # the utility functions assume that channels come last.
    # Since preprocessing step uses channels first, we must adapt to match
    img = tf.constant(np.moveaxis(img, 0, 2), dtype=img.dtype)

    train_data = tf.constant([np.moveaxis(img, 0, 2) for img in train_data],
                             dtype=img.dtype)
    val_data = tf.constant([np.moveaxis(img, 0, 2) for img in val_data],
                           dtype=img.dtype)

    allcnn = make_allcnn()
    allp4cnn = make_allp4cnn()
    allp4mcnn = make_allp4mcnn()

    models = [allcnn, allp4cnn, allp4mcnn]
    for model in models:

        model.compile(
            optimizer=optimizers.SGD(
                learning_rate=0.05,
                momentum=0.9,
                weight_decay=0.001
            ),
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[metrics.sparse_categorical_accuracy],
        )

        history = model.fit(
            train_data,
            train_labels,
            batch_size=128,
            epochs=10,
            callbacks=[callbacks.LearningRateScheduler(scheduler)],
            validation_split=1/5,
            validation_batch_size=1000,
        )

        result = model.evaluate(val_data, val_labels)

    return (allcnn, allp4cnn, allp4mcnn)


def scheduler(epoch, lr):
    if epoch in (200, 250, 300):
        return lr * 0.1
    else:
        return lr


if __name__ == "__main__":
    models = run_cifar10()
    for model in models:
        model.save(model + ".keras")
        del model
