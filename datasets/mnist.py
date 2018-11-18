import numpy as np
from tensorflow.python.keras.datasets import mnist
from scipy.misc import imresize


def load_data(phase='train',
              target_size=(32, 32),
              normalization=None,
              with_label=False):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    assert phase in ['train', 'test']

    x = x_train if phase == 'train' else x_test
    y = y_train if phase == 'train' else y_test

    if target_size is not None or target_size != (28, 28):
        x = np.array([imresize(arr, target_size) for arr in x])

    if normalization is not None:
        x = x.astype('float32')
        if normalization == 'sigmoid':
            x /= 255
        elif normalization == 'tanh':
            x = (x/255 - 0.5) * 2
        else:
            raise ValueError

    x = np.expand_dims(x, -1)

    if with_label:
        return x, y
    else:
        return x
