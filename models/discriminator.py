import tensorflow as tf
from models.base import Discriminator
from ops.blocks import ConvBlock, DenseBlock, ResidualBlock


class MNISTDiscriminator(Discriminator):
    def __init__(self, nb_filter=16,
                 normalization='batch',
                 downsampling='stride',
                 spectral_norm=False):
        super().__init__(nb_filter,
                         normalization,
                         downsampling,
                         spectral_norm)

        self.convs = []
        for i in range(4):
            _nb_filter = nb_filter*(2**i)
            self.convs.append(ConvBlock(_nb_filter,
                                        kernel_size=(5, 5),
                                        sampling=downsampling,
                                        **self.conv_block_params))
        self.dense = DenseBlock(1, spectral_norm=spectral_norm)

    def call(self, inputs,
             training=None,
             mask=None,
             with_feature=False):
        x = inputs
        for conv in self.convs:
            x = conv(x, training=training)
        x = tf.keras.layers.Flatten()(x)
        feature_vector = x

        x = self.dense(x, training=training)

        if with_feature:
            return x, feature_vector
        else:
            return x


class AnimeDiscriminator64(Discriminator):
    def __init__(self, nb_filter=16,
                 normalization='batch',
                 downsampling='stride',
                 spectral_norm=False):
        super().__init__(nb_filter,
                         normalization,
                         downsampling,
                         spectral_norm)

        self.convs = []
        for i in range(5):
            _nb_filter = nb_filter*(2**i)
            self.convs.append(ConvBlock(_nb_filter,
                                        kernel_size=(3, 3),
                                        **self.conv_block_params))
            self.convs.append(ConvBlock(_nb_filter,
                                        kernel_size=(3, 3),
                                        sampling=downsampling,
                                        **self.conv_block_params))
        self.dense = DenseBlock(1, spectral_norm=spectral_norm)

    def call(self, inputs,
             training=None,
             mask=None,
             with_feature=False):
        x = inputs
        for conv in self.convs:
            x = conv(x, training=training)
        x = tf.keras.layers.Flatten()(x)
        feature_vector = x

        x = self.dense(x, training=training)

        if with_feature:
            return x, feature_vector
        else:
            return x


class AnimeDiscriminator128(Discriminator):
    def __init__(self, nb_filter=16,
                 normalization='batch',
                 downsampling='stride',
                 spectral_norm=False):
        super().__init__(nb_filter,
                         normalization,
                         downsampling,
                         spectral_norm)

        self.convs = []
        for i in range(6):
            _nb_filter = nb_filter*(2**i)
            self.convs.append(ConvBlock(_nb_filter,
                                        kernel_size=(3, 3),
                                        **self.conv_block_params))
            self.convs.append(ConvBlock(_nb_filter,
                                        kernel_size=(3, 3),
                                        sampling=downsampling,
                                        **self.conv_block_params))
        self.dense = DenseBlock(1, spectral_norm=spectral_norm)

    def call(self, inputs,
             training=None,
             mask=None,
             with_feature=False):
        x = inputs
        for conv in self.convs:
            x = conv(x, training=training)
        x = tf.keras.layers.Flatten()(x)
        feature_vector = x

        x = self.dense(x, training=training)

        if with_feature:
            return x, feature_vector
        else:
            return x


class AnimeResidualDiscriminator(Discriminator):
    def __init__(self, nb_filter=32,
                 normalization='instance',
                 downsampling='stride',
                 spectral_norm=False):
        super().__init__(nb_filter,
                         normalization,
                         downsampling,
                         spectral_norm)
        self.convs = []
        nb_convs = [1, 2, 2, 2, 2, 2]

        for block_idx, nb_conv in enumerate(nb_convs):
            _nb_filter = nb_filter*(2**block_idx)
            self.convs.append(ConvBlock(_nb_filter,
                                        sampling=self.downsampling,
                                        **self.conv_block_params))
            for _ in range(nb_conv):
                self.convs.append(ResidualBlock(_nb_filter,
                                                **self.conv_block_params))

        self.dense = DenseBlock(1, spectral_norm=spectral_norm)

    def call(self, inputs,
             training=None,
             mask=None,
             with_feature=False):
        x = inputs
        for conv in self.convs:
            x = conv(x, training=training)
        x = tf.keras.layers.Flatten()(x)
        feature_vector = x

        x = self.dense(x, training=training)

        if with_feature:
            return x, feature_vector
        else:
            return x



