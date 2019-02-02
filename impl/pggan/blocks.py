import tensorflow as tf
import sys
sys.path.append('../../')
from ops.blocks import ConvBlock, DenseBlock
from ops.layers.gan import MiniBatchStddev


class FirstGeneratorBlock(tf.keras.Model):
    def __init__(self, filters,
                 normalization='pixel',
                 activation_='lrelu'):
        super().__init__()
        self.conv1 = ConvBlock(filters,
                               sampling='same',
                               normalization=normalization,
                               activation_=activation_,
                               lr_equalization=True)
        self.conv2 = ConvBlock(filters,
                               sampling='same',
                               normalization=normalization,
                               activation_=activation_,
                               lr_equalization=True)

    def call(self, inputs,
             training=None,
             mask=None):
        in_ch = inputs.get_shape().as_list()[-1]
        x = tf.reshape(inputs, (-1, 1, 1, in_ch))
        x = tf.keras.layers.UpSampling2D((4, 4))(x)
        x = self.conv1(x, training=training)
        return self.conv2(x, training=training)


class GeneratorBlock(tf.keras.Model):
    def __init__(self, filters,
                 normalization='pixel',
                 upsampling='subpixel',
                 activation_='lrelu'):
        super().__init__()
        if upsampling in ['subpixel', 'deconv']:
            self.up = ConvBlock(filters,
                                sampling=upsampling,
                                normalization=normalization,
                                activation_=activation_,
                                lr_equalization=True)
            self.feed_training = True
        elif upsampling == 'up':
            self.up = tf.keras.layers.UpSampling2D()
            self.feed_training = False
        else:
            raise ValueError

        self.conv1 = ConvBlock(filters,
                               sampling='same',
                               normalization=normalization,
                               activation_=activation_,
                               lr_equalization=True)
        self.conv2 = ConvBlock(filters,
                               sampling='same',
                               normalization=normalization,
                               activation_=activation_,
                               lr_equalization=True)

    def call(self, inputs,
             training=None,
             mask=None):
        if self.feed_training:
            x = self.up(inputs, training=training)
        else:
            x = self.up(inputs)
        x = self.conv1(x, training=training)
        return self.conv2(x, training=training)


class ToRGB(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = ConvBlock(3,
                              kernel_size=(1, 1),
                              activation_='tanh',
                              lr_equalization=True)

    def call(self, inputs,
             training=None,
             mask=None):
        return self.conv(inputs, training=training)


class DiscriminatorBlock(tf.keras.Model):
    def __init__(self, filters,
                 normalization='instance',
                 downsampling='stride',
                 activation_='lrelu'):
        super().__init__()
        self.conv1 = ConvBlock(filters,
                               sampling='same',
                               normalization=normalization,
                               activation_=activation_,
                               lr_equalization=True)
        self.conv2 = ConvBlock(filters,
                               sampling=downsampling,
                               normalization=normalization,
                               activation_=activation_,
                               lr_equalization=True)

    def call(self, inputs,
             training=None,
             mask=None):
        x = self.conv1(inputs, training=training)
        return self.conv2(x, training=training)


class LastDiscriminatorBlock(tf.keras.Model):
    def __init__(self, filters,
                 normalization='instance',
                 activation_='lrelu'):
        super().__init__()
        self.mb_stddev = MiniBatchStddev()
        self.conv1 = ConvBlock(filters,
                               sampling='same',
                               normalization=normalization,
                               activation_=activation_,
                               lr_equalization=True)
        self.conv2 = ConvBlock(filters,
                               kernel_size=(4, 4),
                               sampling='same',
                               padding='valid',
                               normalization='layer',
                               activation_=activation_,
                               lr_equalization=True)

    def call(self, inputs,
             training=None,
             mask=None):
        x = self.mb_stddev(inputs)
        x = self.conv1(x, training=training)
        return self.conv2(x, training=training)


class FromRGB(tf.keras.Model):
    def __init__(self, filters,
                 normalization='instance',
                 activation_='lrelu'):
        super().__init__()
        self.conv = ConvBlock(filters,
                              kernel_size=(1, 1),
                              sampling='same',
                              normalization=normalization,
                              activation_=activation_,
                              lr_equalization=True)

    def call(self, inputs,
             training=None,
             mask=None):
        return self.conv(inputs, training=training)
