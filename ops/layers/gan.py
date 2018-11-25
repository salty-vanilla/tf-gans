import tensorflow as tf
from tensorflow.python.keras import layers as kl
from tensorflow.contrib import layers as tl
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
import numpy as np


class MiniBatchStddev(Layer):
    def __init__(self, group_size=1, **kwargs):
        self.group_size = group_size
        super(MiniBatchStddev, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = tf.keras.layers.InputSpec('float32', input_shape)

    def call(self, x, *args, **kwargs):
        _, h, w, c = self.input_spec.shape
        # gs = K.maximum(self.group_size, self.input_spec.shape[0])
        gs = self.group_size
        _x = K.reshape(x, (gs, -1, h, w, c))
        _x -= K.mean(_x, axis=0, keepdims=True)
        _x = K.mean(K.square(_x), axis=0)
        _x = K.sqrt(_x + K.epsilon())
        _x = K.sum(_x, axis=[1, 2, 3], keepdims=True)
        # _x = K.tile(x, [gs, h, w, 1])
        _x = tf.tile(_x, [gs, h, w, 1])
        _x = K.concatenate([x, _x], axis=-1)
        return _x

    def compute_output_shape(self, input_shape):
        return (*input_shape[:3], input_shape[3]+1)


class MiniBatchDiscrimination(Layer):
    def __init__(self, nb_kernel=100,
                 dim_per_kernel=5,
                 trainable=True,
                 **kwargs):
        self.nb_kernel = nb_kernel
        self.dim_per_kernel = dim_per_kernel
        self.trainable = trainable
        super(MiniBatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        self.T = K.variable(np.random.normal(size=(input_shape[1],
                                                   self.nb_kernel*self.dim_per_kernel)),
                            name='MBD_T')
        super(MiniBatchDiscrimination, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        _x = K.dot(x, self.T)
        _x = K.reshape(_x, shape=(-1, self.nb_kernel, self.dim_per_kernel))
        diffs = K.expand_dims(_x, 3) - K.expand_dims(K.permute_dimensions(_x, [1, 2, 0]), 0)
        abs_diffs = K.sum(K.abs(diffs), 2)
        _x = K.sum(K.exp(-abs_diffs), 2)
        return K.concatenate([x, _x], axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]+self.nb_kernel


class LearningRateEqualizer(tf.keras.layers.Layer):
    def __init__(self, layer):
        super().__init__(layer)
        self.layer = layer
        self._is_set = False

    def call(self,
             inputs,
             training=None,
             **kwargs):
        if training:
            if not self._is_set:
                self.layer(inputs)
                self._is_set = True
                c = np.array(tf.reduce_mean(self.layer.kernel**2)**0.5)
                self.scale = self.add_variable(name='scale',
                                               shape=(),
                                               initializer=tf.keras.initializers.Constant(c),
                                               trainable=False)
            tf.assign(self.layer.kernel, self.layer.kernel/self.scale)
        else:
            pass
        return self.layer(inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape
