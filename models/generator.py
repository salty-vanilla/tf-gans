import tensorflow as tf
from models.base import Generator
from ops.blocks import ConvBlock, DenseBlock, ResidualBlock


class MNISTGenerator(Generator):
    def __init__(self, latent_dim,
                 nb_filter=16,
                 last_activation='tanh',
                 normalization='batch',
                 upsampling='deconv',
                 spectral_norm=False):
        super().__init__(latent_dim,
                         nb_filter,
                         last_activation,
                         normalization,
                         upsampling,
                         spectral_norm)

        self.convs = []
        self.dense = DenseBlock(4*4*nb_filter*(2**3),
                                activation_='relu',
                                normalization=normalization,
                                spectral_norm=spectral_norm)

        for i in range(1, 4):
            _nb_filter = nb_filter*(2**(3-i))

            if upsampling == 'subpixel':
                _nb_filter *= 4
            self.convs.append(ConvBlock(_nb_filter,
                                        kernel_size=(5, 5),
                                        sampling=upsampling,
                                        **self.conv_block_params))
        self.last_conv = ConvBlock(1,
                                   kernel_size=(1, 1),
                                   **self.last_conv_block_params)

    def call(self, inputs,
             training=None,
             mask=None):
        x = self.dense(inputs, training=training)
        x = tf.reshape(x, (-1, 4, 4, self.nb_filter*(2**3)))
        for conv in self.convs:
            x = conv(x, training=training)
        x = self.last_conv(x, training=training)
        return x


class AnimeGenerator64(Generator):
    def __init__(self, latent_dim,
                 nb_filter=16,
                 last_activation='tanh',
                 normalization='batch',
                 upsampling='deconv',
                 spectral_norm=False):
        super().__init__(latent_dim,
                         nb_filter,
                         last_activation,
                         normalization,
                         upsampling,
                         spectral_norm)

        self.convs = []
        self.dense = DenseBlock(4*4*nb_filter*(2**4),
                                activation_='relu',
                                normalization=normalization,
                                spectral_norm=spectral_norm)

        for i in range(1, 5):
            _nb_filter = nb_filter*(2**(4-i))
            if upsampling == 'subpixel':
                _nb_filter *= 4
            self.convs.append(ConvBlock(_nb_filter,
                                        kernel_size=(5, 5),
                                        sampling=upsampling,
                                        **self.conv_block_params))
        self.last_conv = ConvBlock(3,
                                   kernel_size=(7, 7),
                                   **self.last_conv_block_params)

    def call(self, inputs,
             training=None,
             mask=None):
        x = self.dense(inputs, training=training)
        x = tf.reshape(x, (-1, 4, 4, self.nb_filter*(2**4)))
        for conv in self.convs:
            x = conv(x, training=training)
        x = self.last_conv(x, training=training)
        return x


class AnimeGenerator128(Generator):
    def __init__(self, latent_dim,
                 nb_filter=16,
                 last_activation='tanh',
                 normalization='batch',
                 upsampling='deconv',
                 spectral_norm=False):
        super().__init__(latent_dim,
                         nb_filter,
                         last_activation,
                         normalization,
                         upsampling,
                         spectral_norm)

        self.convs = []
        self.dense = DenseBlock(4*4*nb_filter*(2**5),
                                activation_='relu',
                                normalization=normalization,
                                spectral_norm=spectral_norm)

        for i in range(1, 6):
            _nb_filter = nb_filter*(2**(5-i))
            if upsampling == 'subpixel':
                _nb_filter *= 4
            self.convs.append(ConvBlock(_nb_filter,
                                        kernel_size=(5, 5),
                                        sampling=upsampling,
                                        **self.conv_block_params))
        self.last_conv = ConvBlock(3,
                                   kernel_size=(9, 9),
                                   **self.last_conv_block_params)

    def call(self, inputs,
             training=None,
             mask=None):
        x = self.dense(inputs, training=training)
        x = tf.reshape(x, (-1, 4, 4, self.nb_filter*(2**5)))
        for conv in self.convs:
            x = conv(x, training=training)
        x = self.last_conv(x, training=training)
        return x


class SRResnetGenerator(Generator):
    def __init__(self, latent_dim,
                 nb_filter=64,
                 last_activation='tanh',
                 normalization='batch',
                 upsampling='deconv',
                 spectral_norm=False):
        super().__init__(latent_dim,
                         nb_filter,
                         last_activation,
                         normalization,
                         upsampling,
                         spectral_norm)

        self.dense = DenseBlock(16*16*nb_filter,
                                activation_='lrelu',
                                normalization=normalization,
                                spectral_norm=spectral_norm)
        self.convs = [ResidualBlock(nb_filter,
                                    kernel_size=(3, 3),
                                    **self.conv_block_params)
                      for _ in range(5)]
        if self.upsampling == 'subpixel':
            nb_filter *= 4
        self.ups = [ConvBlock(nb_filter,
                              kernel_size=(3, 3),
                              sampling=self.upsampling,
                              **self.conv_block_params)
                    for _ in range(3)]

        self.last_conv = ConvBlock(3,
                                   kernel_size=(9, 9),
                                   **self.last_conv_block_params)

    def call(self, inputs,
             training=None,
             mask=None):
        x = self.dense(inputs, training=training)
        x = tf.reshape(x, (-1, 16, 16, self.nb_filter))
        h = x
        for conv in self.convs:
            x = conv(x, training=training)
        x += h
        for up in self.ups:
            x = up(x, training=training)
        x = self.last_conv(x, training=training)
        return x