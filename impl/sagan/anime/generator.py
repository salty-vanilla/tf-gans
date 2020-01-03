import tensorflow as tf
import os
import sys
sys.path.append(os.getcwd())
from models.base import Generator as G
from ops.blocks import ConvBlock, DenseBlock, ResidualBlock
from ops.layers.non_local import NonLocal2D


class AnimeGenerator64(G):
    def __init__(self, latent_dim,
                 nb_filter=16,
                 last_activation='tanh',
                 normalization='batch',
                 upsampling='deconv',
                 spectral_norm=True):
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
                                        spectral_norm=spectral_norm,
                                        **self.conv_block_params))
        self.last_conv = ConvBlock(3,
                                   kernel_size=(7, 7),
                                   spectral_norm=spectral_norm,
                                   **self.last_conv_block_params)
        self.non_local = NonLocal2D(nb_filter*(2**(4-2)))

    def call(self, inputs,
             training=None,
             mask=None):
        x = self.dense(inputs, training=training)
        x = tf.reshape(x, (-1, 4, 4, self.nb_filter*(2**4)))
        for i, conv in enumerate(self.convs):
            x = conv(x, training=training)

            if i == 2:
                x = self.non_local(x, training=training)

        x = self.last_conv(x, training=training)
        return x


class ResidualAnimeGenerator64(G):
    def __init__(self, latent_dim,
                 nb_filter=16,
                 last_activation='tanh',
                 normalization='batch',
                 upsampling='deconv',
                 spectral_norm=True):
        super().__init__(latent_dim,
                         nb_filter,
                         last_activation,
                         normalization,
                         upsampling,
                         spectral_norm)

        self.convs = []
        self.dense = DenseBlock(4*4*nb_filter*(2**4),
                                activation_='lrelu',
                                normalization=normalization,
                                spectral_norm=spectral_norm)

        for i in range(1, 5):
            _nb_filter = nb_filter*(2**(4-i))
            if upsampling == 'subpixel':
                _nb_filter *= 4
            self.convs.append(ResidualBlock(_nb_filter,
                                            kernel_size=(3, 3),
                                            sampling=upsampling,
                                            **self.conv_block_params))
        self.last_conv = ConvBlock(3,
                                   kernel_size=(3, 3),
                                   **self.last_conv_block_params)
        self.non_local = NonLocal2D(nb_filter*(2**(4-3)))

    def call(self, inputs,
             training=None,
             mask=None):
        x = self.dense(inputs, training=training)
        x = tf.reshape(x, (-1, 4, 4, self.nb_filter*(2**4)))
        for i, conv in enumerate(self.convs):
            x = conv(x, training=training)

            if i == 2:
                x = self.non_local(x, training=training)

        x = self.last_conv(x, training=training)
        return x


class DeepResidualAnimeGenerator64(G):
    def __init__(self, latent_dim,
                 nb_filter=16,
                 last_activation='tanh',
                 normalization='batch',
                 upsampling='deconv',
                 spectral_norm=True):
        super().__init__(latent_dim,
                         nb_filter,
                         last_activation,
                         normalization,
                         upsampling,
                         spectral_norm)

        self.up_convs = []
        self.first_up = ConvBlock(1024, 
                                  kernel_size=(16, 16),
                                  sampling='deconv',
                                  activation_='lrelu',
                                  normalization='batch',
                                  padding='VALID')

        for _ in range(2):
            _nb_filter = 64
            if upsampling == 'subpixel': _nb_filter *= 4
            self.up_convs.append(ResidualBlock(_nb_filter,
                                               kernel_size=(3, 3),
                                               sampling=upsampling,
                                               **self.conv_block_params))
        self.last_conv = ConvBlock(3,
                                   kernel_size=(9, 9),
                                   **self.last_conv_block_params)

        self.convs = [ResidualBlock(64, 
                                    kernel_size=(3, 3),
                                    sampling='same',
                                    **self.conv_block_params)
                      for _ in range(5)]
        self.non_locals = [NonLocal2D(64) for _ in range(5)]
        
        self.conv1 = ConvBlock(64, 
                               kernel_size=(1, 1),
                               sampling='same',
                               spectral_norm=spectral_norm,
                               activation_=None,
                               normalization=None)
        self.conv2 = ConvBlock(64, 
                               kernel_size=(1, 1),
                               sampling='same',
                               spectral_norm=spectral_norm,
                               activation_=None,
                               normalization=None)
        self.conv3 = ConvBlock(64, 
                               kernel_size=(1, 1),
                               sampling='same',
                               spectral_norm=spectral_norm,
                               activation_=None,
                               normalization=None)
        self.conv4 = ConvBlock(64, 
                               kernel_size=(1, 1),
                               sampling='same',
                               spectral_norm=spectral_norm,
                               activation_=None,
                               normalization=None)
        self.last_modify_blocks = [ResidualBlock(64, 
                                                 kernel_size=(3, 3),
                                                 sampling='same',
                                                 **self.conv_block_params)
                                   for _ in range(3)]

    def call(self, inputs,
             training=None,
             mask=None):
        # 1st block
        x = inputs
        x = tf.expand_dims(x, axis=1)
        x = tf.expand_dims(x, axis=1)
        x = self.first_up(x, training=training)

        # 2nd block
        sc = self.conv1(x)
        for conv, nl in zip(self.convs, self.non_locals):
            x = conv(x, training=training)
            x = nl(x, training=training)
        x = self.conv2(x)
        x += sc

        # 3rd block
        for conv in self.up_convs:
            x = conv(x, training=training)

        # 4th block        
        sc = self.conv3(x)
        for conv in self.last_modify_blocks:
            x = conv(x, training=training)
        x = self.conv4(x)
        x += sc
        x = self.last_conv(x, training=training)
        return x


class DeepResidualAnimeGenerator128(G):
    def __init__(self, latent_dim,
                 nb_filter=16,
                 last_activation='tanh',
                 normalization='batch',
                 upsampling='deconv',
                 spectral_norm=True):
        super().__init__(latent_dim,
                         nb_filter,
                         last_activation,
                         normalization,
                         upsampling,
                         spectral_norm)

        self.up_convs = []
        self.first_up = ConvBlock(1024, 
                                  kernel_size=(16, 16),
                                  sampling='deconv',
                                  activation_='lrelu',
                                  normalization='batch',
                                  padding='VALID',
                                  spectral_norm=spectral_norm)

        for _ in range(3):
            _nb_filter = 64
            if upsampling == 'subpixel': _nb_filter *= 4
            self.up_convs.append(ResidualBlock(_nb_filter,
                                               kernel_size=(3, 3),
                                               sampling=upsampling,
                                               **self.conv_block_params))
        self.last_conv = ConvBlock(3,
                                   kernel_size=(9, 9),
                                   **self.last_conv_block_params)

        self.convs = [ResidualBlock(64, 
                                    kernel_size=(3, 3),
                                    sampling='same',
                                    **self.conv_block_params)
                      for _ in range(16)]
        self.non_locals = [NonLocal2D(64) for _ in range(16)]
        
        self.conv1 = ConvBlock(64, 
                               kernel_size=(1, 1),
                               sampling='same',
                               spectral_norm=spectral_norm,
                               activation_=None,
                               normalization=None)
        self.conv2 = ConvBlock(64, 
                               kernel_size=(1, 1),
                               sampling='same',
                               spectral_norm=spectral_norm,
                               activation_=None,
                               normalization=None)
        self.conv3 = ConvBlock(64, 
                               kernel_size=(1, 1),
                               sampling='same',
                               spectral_norm=spectral_norm,
                               activation_=None,
                               normalization=None)
        self.conv4 = ConvBlock(64, 
                               kernel_size=(1, 1),
                               sampling='same',
                               spectral_norm=spectral_norm,
                               activation_=None,
                               normalization=None)
        self.last_modify_blocks = [ResidualBlock(64, 
                                                 kernel_size=(3, 3),
                                                 sampling='same',
                                                 **self.conv_block_params)
                                   for _ in range(3)]

    def call(self, inputs,
             training=None,
             mask=None):
        # 1st block
        x = inputs
        x = tf.expand_dims(x, axis=1)
        x = tf.expand_dims(x, axis=1)
        x = self.first_up(x, training=training)

        # 2nd block
        sc = self.conv1(x)
        for conv, nl in zip(self.convs, self.non_locals):
            x = conv(x, training=training)
            x = nl(x, training=training)
        x = self.conv2(x)
        x += sc

        # 3rd block
        for conv in self.up_convs:
            x = conv(x, training=training)

        # 4th block        
        sc = self.conv3(x)
        for conv in self.last_modify_blocks:
            x = conv(x, training=training)
        x = self.conv4(x)
        x += sc
        x = self.last_conv(x, training=training)
        return x