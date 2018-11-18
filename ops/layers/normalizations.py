import tensorflow as tf
import tensorflow.keras.backend as K


class PixelNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PixelNorm, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = inputs
        return x / tf.sqrt(tf.reduce_mean(tf.square(x),
                                          axis=-1,
                                          keepdims=True))

    def compute_output_shape(self, input_shape):
        return input_shape


class InstanceNorm(tf.keras.layers.Layer):
    def __init__(self, beta_initializer='zeros',
                 gamma_initializer='ones'):
        super().__init__()
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(input_shape[-1]),
                                     name='gamma',
                                     initializer=self.gamma_initializer)
        self.beta = self.add_weight(shape=(input_shape[-1]),
                                    name='beta',
                                    initializer=self.beta_initializer)

    def call(self, inputs, **kwargs):
        x = inputs
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        return tf.nn.batch_normalization(x,
                                         mean,
                                         var,
                                         offset=self.beta,
                                         scale=self.gamma,
                                         variance_epsilon=K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape


class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, begin_norm_axis=1,
                 begin_params_axis=-1,
                 beta_initializer='zeros',
                 gamma_initializer='ones'):
        super().__init__()
        self.begin_norm_axis = begin_norm_axis
        self.begin_param_axis = begin_params_axis
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer

    def build(self, input_shape):
        params_shape = input_shape[self.begin_param_axis:]
        self.input_rank = input_shape.ndims
        if self.begin_norm_axis < 0:
            self.begin_norm_axis = self.input_rank + self.begin_norm_axis

        self.gamma = self.add_weight(shape=params_shape,
                                     name='gamma',
                                     initializer=self.gamma_initializer)
        self.beta = self.add_weight(shape=params_shape,
                                    name='beta',
                                    initializer=self.beta_initializer)

    def call(self, inputs, **kwargs):
        x = inputs
        norm_axes = list(range(self.begin_norm_axis, self.input_rank))
        mean, var = tf.nn.moments(inputs, norm_axes, keep_dims=True)
        return tf.nn.batch_normalization(x,
                                         mean,
                                         var,
                                         offset=self.beta,
                                         scale=self.gamma,
                                         variance_epsilon=K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape
