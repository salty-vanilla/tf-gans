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


class SpectralNorm(tf.keras.layers.Layer):
    def __init__(self, layer,
                 power_iteration=1,
                 initializer=tf.keras.initializers.TruncatedNormal(stddev=0.2)):
        super().__init__()
        self.layer = layer
        self.power_iteration = power_iteration
        self.initializer = initializer
        self._is_set = False

    def build(self, input_shape):
        if hasattr(self.layer, 'filters'):
            d = self.layer.filters
        elif hasattr(self.layer, 'units'):
            d = self.layer.units
        else:
            raise AttributeError
        self.u = self.add_variable(shape=(1, d),
                                   name='u',
                                   initializer=self.initializer,
                                   trainable=False)

    def call(self, inputs,
             training=None,
             **kwargs):
        if training:
            if not self._is_set:
                self.layer(inputs)
                self._is_set = True

            w = self.layer.kernel
            w_shape = w.shape.as_list()
            w = tf.reshape(w, (-1, w_shape[-1]))

            u_hat = self.u
            v_hat = None

            for i in range(self.power_iteration):
                v_ = tf.matmul(u_hat, w, transpose_b=True)
                v_hat = tf.nn.l2_normalize(v_)

                u_ = tf.matmul(v_hat, w)
                u_hat = tf.nn.l2_normalize(u_)

            u_hat = tf.stop_gradient(u_hat)
            v_hat = tf.stop_gradient(v_hat)

            sigma = tf.matmul(tf.matmul(v_hat, w), u_hat, transpose_b=True)

            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, w_shape)

            tf.assign(self.u, u_hat)
            tf.assign(self.layer.kernel, w_norm)
        else:
            pass
        return self.layer(inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape
