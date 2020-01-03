import tensorflow as tf
import os
import sys
sys.path.append(os.getcwd())
from models import Discriminator as D
from ops.blocks import ConvBlock, DenseBlock, ResidualBlock
from ops.layers.non_local import NonLocal2D


class Discriminator(D):
    def __init__(self, nb_filter=16,
                 normalization='batch',
                 downsampling='stride',
                 spectral_norm=True):
        super().__init__(nb_filter,
                         normalization,
                         downsampling)

        self.convs = []
        for i in range(5):
            _nb_filter = nb_filter*(2**i)
            self.convs.append(ConvBlock(_nb_filter,
                                        kernel_size=(5, 5),
                                        sampling=downsampling,
                                        spectral_norm=spectral_norm,
                                        **self.conv_block_params))
        self.dense = DenseBlock(1, spectral_norm=spectral_norm)
        self.non_local = NonLocal2D(nb_filter*(2**(2)))

    def call(self, inputs,
             training=None,
             mask=None,
             with_feature=False):
        x = inputs
        for i, conv in enumerate(self.convs):
            x = conv(x, training=training)

            if i == 2:
                x = self.non_local(x, training=training)

        x = tf.keras.layers.Flatten()(x)
        feature_vector = x

        x = self.dense(x, training=training)

        if with_feature:
            return x, feature_vector
        else:
            return x
