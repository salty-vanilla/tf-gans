import tensorflow as tf
from ops.layers import activation as act


class SubPixelConv2D(tf.keras.layers.Conv2D):
    def __init__(self, filters,
                 rate=2,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None):
        super().__init__(filters,
                         kernel_size,
                         strides,
                         padding,
                         data_format,
                         dilation_rate,
                         None,
                         use_bias,
                         kernel_initializer,
                         bias_initializer,
                         kernel_regularizer,
                         bias_regularizer,
                         activity_regularizer,
                         kernel_constraint,
                         bias_constraint)
        self.activation = activation
        self.rate = rate

    def call(self, inputs,
             *args,
             **kwargs):
        x = inputs
        x = super().call(x)
        if self.activation:
            x = act(x, self.activation)
        return tf.depth_to_space(x, self.rate)

    def compute_output_shape(self, input_shape):
        return input_shape[0], \
               input_shape[1]*self.rate, \
               input_shape[2]*self.rate, \
               self.filters//(self.rate**2)
