import tensorflow as tf
from blocks import DiscriminatorBlock, LastDiscriminatorBlock, FromRGB
import sys
sys.path.append('../../')
from ops.blocks import DenseBlock


class Discriminator(tf.keras.Model):
    def __init__(self, nb_growing=8,
                 downsampling='stride',
                 normalization='instance',
                 activation_='lrelu'):
        super().__init__()
        self.filters = [512, 512, 512, 512, 256, 256, 128, 64, 32][:nb_growing]

        self.blocks = []
        self.from_rgbs = []

        for i in range(nb_growing):
            if i == 0:
                self.blocks.append(LastDiscriminatorBlock(self.filters[i],
                                                          normalization=normalization,
                                                          activation_=activation_))
            else:
                self.blocks.append(DiscriminatorBlock(self.filters[i-1],
                                                      normalization=normalization,
                                                      downsampling=downsampling,
                                                      activation_=activation_))
            self.from_rgbs.append(FromRGB(self.filters[i],
                                          normalization=normalization,
                                          activation_=activation_))
        self.dense = DenseBlock(1,
                                lr_equalization=True)

    def call(self, inputs,
             training=None,
             mask=None,
             growing_step: int=None,
             alpha: float=1.):
        assert isinstance(growing_step, int)

        x = inputs
        is_blend = alpha != 1. and growing_step != 0
        if is_blend:
            _x = tf.keras.layers.AveragePooling2D((2, 2))(x)
            _x = self.from_rgbs[growing_step-1](_x)

        x = self.from_rgbs[growing_step](x)
        for i in range(growing_step+1)[::-1]:
            x = self.blocks[i](x)

            if is_blend and i == growing_step:
                x = alpha*x + (1.-alpha)*_x
        x = tf.keras.layers.Flatten()(x)
        return self.dense(x, training=training)


if __name__ == '__main__':
    tf.enable_eager_execution()
    d = Discriminator(nb_growing=8)
    for i_ in range(8):
        x_ = tf.random_normal(shape=(5, 4*(2**i_), 4*(2**i_), 3))
        print(x_.shape)
        print(d(x_, alpha=0.5, growing_step=i_, training=True).shape)
