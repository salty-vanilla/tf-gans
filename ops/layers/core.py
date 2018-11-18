import tensorflow as tf


class Padding(tf.keras.layers.Layer):
    def __init__(self, size,
                 mode='constant',
                 constant_values=0):
        super().__init__()
        self.size = size
        self.constant_values = constant_values

        if mode in ['CONSTANT', 'constant', 'zero', 'ZERO']:
            self.mode = 'CONSTANT'
        elif mode in ['REFLECT', 'reflect']:
            self.mode = 'REFLECT'
        elif mode in ['SYMMETRIC', 'symmetric']:
            self.mode = 'SYMMETRIC'
        else:
            raise ValueError

    def call(self, inputs, **kwargs):
        x = inputs
        return tf.pad(x,
                      [[0, 0],
                       [self.size[0], self.size[0]],
                       [self.size[1], self.size[1]],
                       [0, 0]],
                      mode=self.mode,
                      constant_values=self.constant_values)
