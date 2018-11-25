import tensorflow as tf
from blocks import FirstGeneratorBlock, GeneratorBlock, ToRGB


class Generator(tf.keras.Model):
    def __init__(self, nb_growing=8,
                 upsampling='subpixel',
                 normalization='pixel',
                 activation_='lrelu'):
        super().__init__()
        self.filters = [512, 512, 512, 512, 256, 256, 128, 64, 32][:nb_growing]

        self.blocks = []
        self.to_rgbs = []

        for i in range(nb_growing):
            if i == 0:
                self.blocks.append(FirstGeneratorBlock(filters=self.filters[i],
                                                       normalization=normalization,
                                                       activation_=activation_))
            else:
                self.blocks.append(GeneratorBlock(filters=self.filters[i],
                                                  normalization=normalization,
                                                  upsampling=upsampling,
                                                  activation_=activation_))
            self.to_rgbs.append(ToRGB())

    def call(self, inputs,
             training=None,
             mask=None,
             growing_step: int=None,
             alpha: float=1.):
        assert isinstance(growing_step, int)

        is_blend = alpha != 1. and growing_step != 0

        x = inputs
        for i in range(growing_step + 1):
            x = self.blocks[i](x)

            if is_blend and i == growing_step-1:
                _x = tf.keras.layers.UpSampling2D((2, 2))(x)
                _x = self.to_rgbs[growing_step-1](_x)
        x = self.to_rgbs[growing_step](x)

        if is_blend:
            x = alpha*x + (1.-alpha)*_x

        return x


if __name__ == '__main__':
    tf.enable_eager_execution()
    z_ = tf.random_normal(shape=(1, 100))
    g = Generator(nb_growing=8)
    for i_ in range(8):
        print(g(z_, alpha=0.5, growing_step=i_, training=True).shape)
