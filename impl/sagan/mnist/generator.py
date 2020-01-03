import tensorflow as tf
import os
import sys
sys.path.append(os.getcwd())
from models import Generator as G
from ops.blocks import ConvBlock, DenseBlock
from ops.layers.non_local import NonLocal2D


class Generator(G):
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
                         upsampling)

        self.convs = []
        self.dense = DenseBlock(4*4*nb_filter*(2**3),
                                activation_='lrelu',
                                normalization=normalization,
                                spectral_norm=spectral_norm)

        for i in range(4):
            _nb_filter = nb_filter*(2**(3-i))

            if i != 0:
                if upsampling == 'subpixel':
                    _nb_filter *= 4
                self.convs.append(ConvBlock(_nb_filter,
                                            kernel_size=(5, 5),
                                            sampling=upsampling,
                                            spectral_norm=spectral_norm,
                                            **self.conv_block_params))
        self.last_conv = ConvBlock(1,
                                   kernel_size=(1, 1),
                                   spectral_norm=spectral_norm,
                                   **self.last_conv_block_params)
        self.non_local = NonLocal2D(nb_filter*(2**(3-1)))

    def call(self, inputs,
             training=None,
             mask=None):
        x = self.dense(inputs, training=training)
        x = tf.reshape(x, (-1, 4, 4, self.nb_filter*(2**3)))
        for i, conv in enumerate(self.convs):
            x = conv(x, training=training)

            if i == 3-1:
                x = self.non_local(x, training=training)

        x = self.last_conv(x, training=training)
        return x
