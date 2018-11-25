import tensorflow as tf
from ops.layers.activations import activation
from ops.layers.conv import SubPixelConv2D
from ops.layers.normalizations import *
from ops.layers.gan import LearningRateEqualizer


class ConvBlock(tf.keras.Model):
    def __init__(self, filters,
                 kernel_size=(3, 3),
                 activation_=None,
                 dilation_rate=(1, 1),
                 sampling='same',
                 normalization=None,
                 spectral_norm=False,
                 lr_equalization=False,
                 **conv_params):
        conv_params.setdefault('padding', 'same')
        super().__init__()

        stride = 1 if sampling in ['same',
                                   'subpixel',
                                   'max_pool',
                                   'avg_pool',
                                   'subpixel'] \
            else 2
        if 'stride' in conv_params:
            stride = conv_params['stride']

        # Convolution
        if sampling in ['up', 'max_pool', 'avg_pool', 'same', 'stride']:
            s = stride if sampling == 'stride' else 1
            self.conv = tf.keras.layers.Conv2D(filters,
                                               kernel_size,
                                               strides=s,
                                               dilation_rate=dilation_rate,
                                               activation=None,
                                               **conv_params)
        elif sampling == 'deconv':
            self.conv = tf.keras.layers.Conv2DTranspose(filters,
                                                        kernel_size,
                                                        strides=stride,
                                                        dilation_rate=dilation_rate,
                                                        activation=None,
                                                        **conv_params)
        elif sampling == 'subpixel':
            self.conv = SubPixelConv2D(filters,
                                       rate=2,
                                       kernel_size=kernel_size,
                                       activation=None,
                                       **conv_params)
        else:
            raise ValueError

        if spectral_norm:
            self.conv = SpectralNorm(self.conv)

        if lr_equalization:
            self.conv = LearningRateEqualizer(self.conv)

        # Normalization
        if normalization is not None:
            if normalization == 'batch':
                self.norm = tf.keras.layers.BatchNormalization()
            elif normalization == 'layer':
                self.norm = LayerNorm()
            elif normalization == 'instance':
                self.norm = InstanceNorm()
            elif normalization == 'pixel':
                self.norm = PixelNorm()
            else:
                raise ValueError
        else:
            self.norm = None

        self.act = activation_

        self.is_feed_training = spectral_norm or lr_equalization

    def call(self, inputs,
             training=None,
             mask=None):
        if self.is_feed_training:
            x = self.conv(inputs, training=training)
        else:
            x = self.conv(inputs)
        if self.norm is not None:
            x = self.norm(x, training=training)
        x = activation(x, self.act)
        return x


class DenseBlock(tf.keras.Model):
    def __init__(self, units,
                 activation_=None,
                 normalization=None,
                 spectral_norm=False,
                 lr_equalization=False,
                 **dense_params):
        super().__init__()
        self.units = units
        self.dense = tf.keras.layers.Dense(units, **dense_params)

        if spectral_norm:
            self.dense = SpectralNorm(self.dense)

        if lr_equalization:
            self.dense = LearningRateEqualizer(self.dense)

        # Normalization
        if normalization is not None:
            if normalization == 'batch':
                self.norm = tf.keras.layers.BatchNormalization()
            elif normalization == 'layer':
                self.norm = LayerNorm()
            elif normalization == 'instance':
                self.norm = None
            elif normalization == 'pixel':
                self.norm = None
            else:
                raise ValueError
        else:
            self.norm = None

        self.act = activation_

        self.is_feed_training = spectral_norm or lr_equalization

    def call(self, inputs,
             training=None,
             mask=None):
        if self.is_feed_training:
            x = self.dense(inputs, training=training)
        else:
            x = self.dense(inputs)
        if self.norm is not None:
            x = self.norm(x, training=training)
        x = activation(x, self.act)
        return x
