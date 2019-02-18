import tensorflow as tf


class Generator(tf.keras.Model):
    def __init__(self, latent_dim,
                 nb_filter=32,
                 last_activation='tanh',
                 normalization='batch',
                 upsampling='deconv',
                 spectral_norm=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.nb_filter = nb_filter

        self.last_activation = last_activation
        self.normalization = normalization
        self.upsampling = upsampling
        self.spectral_norm = spectral_norm
        self.feature_shape = None

        self.conv_block_params = {'kernel_initializer': 'he_normal',
                                  'activation_': 'relu',
                                  'normalization': self.normalization,
                                  'spectral_norm': self.spectral_norm}

        self.last_conv_block_params = {'kernel_initializer': 'glorot_uniform',
                                       'activation_': self.last_activation,
                                       'normalization': None,
                                       'spectral_norm': self.spectral_norm}


class Discriminator(tf.keras.Model):
    def __init__(self, nb_filter=32,
                 normalization='batch',
                 downsampling='stride',
                 spectral_norm=False):
        super().__init__()
        self.nb_filter = nb_filter
        self.normalization = normalization
        self.downsampling = downsampling
        self.spectral_norm = spectral_norm

        self.conv_block_params = {'kernel_initializer': 'he_normal',
                                  'activation_': 'lrelu',
                                  'normalization': self.normalization,
                                  'spectral_norm': self.spectral_norm}
