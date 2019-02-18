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

        if sampling == 'max_pool':
            self.pool = tf.keras.layers.MaxPool2D()
        elif sampling == 'avg_pool':
            self.pool = tf.keras.layers.AveragePooling2D()
        else:
            self.pool = None

        if sampling == 'up':
            self.up = tf.keras.layers.UpSampling2D()
        else:
            self.up = None

        self.is_feed_training = spectral_norm or lr_equalization

    def call(self, inputs,
             training=None,
             mask=None):
        x = inputs
        if self.up is not None:
            x = self.up(x)
        if self.is_feed_training:
            x = self.conv(x, training=training)
        else:
            x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x, training=training)
        x = activation(x, self.act)
        if self.pool is not None:
            x = self.pool(x)
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


class ResidualBlock(tf.keras.Model):
    def __init__(self, filters,
                 kernel_size=(3, 3),
                 activation_=None,
                 dilation_rate=(1, 1),
                 sampling='same',
                 normalization=None,
                 spectral_norm=False,
                 lr_equalization=False,
                 **conv_params):
        super().__init__()
        self.filters = filters
        self.sampling = sampling
        self.conv1 = ConvBlock(filters,
                               kernel_size,
                               activation_,
                               dilation_rate,
                               'same',
                               normalization,
                               spectral_norm,
                               lr_equalization,
                               **conv_params)
        self.conv2 = ConvBlock(filters,
                               kernel_size,
                               None,
                               dilation_rate,
                               sampling,
                               normalization,
                               spectral_norm,
                               lr_equalization,
                               **conv_params)
        self.act = activation_
        self.shortcut_conv = tf.keras.layers.Conv2D(filters, (1, 1))

    def call(self, inputs,
             training=None,
             mask=None):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)

        if inputs.get_shape().as_list()[-1] != self.filters:
            x += self.shortcut(self.shortcut_conv(inputs))
        else:
            x += self.shortcut(inputs)
        x = activation(x, self.act)
        return x

    def shortcut(self, x):
        if self.sampling in ['deconv', 'up', 'subpixel']:
            return tf.keras.layers.UpSampling2D()(x)
        elif self.sampling in ['max_pool', 'avg_pool', 'stride']:
            return tf.keras.layers.AveragePooling2D()(x)
        else:
            return x
