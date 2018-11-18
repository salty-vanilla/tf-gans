import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import resize_images
from tensorflow.python.ops.image_ops_impl import ResizeMethod


class Resize(tf.keras.layers.Layer):
    def __init__(self, target_size,
                 interpolation='bilinear'):
        super().__init__()
        self.target_size = target_size
        if interpolation in ['bilinear', 'BILINEAR']:
            self.interpolation = ResizeMethod.BILINEAR
        elif interpolation in ['nearest_neighbor', 'NEAREST_NEIGHBOR']:
            self.interpolation = ResizeMethod.NEAREST_NEIGHBOR
        elif interpolation in ['bicubic', 'BICUBIC']:
            self.interpolation = ResizeMethod.BICUBIC
        elif interpolation in ['area', 'AREA']:
            self.interpolation = ResizeMethod.AREA
        else:
            raise ValueError

    def call(self, inputs, **kwargs):
        x = inputs
        return resize_images(x, self.target_size, self.interpolation)

    def compute_output_shape(self, input_shape):
        return input_shape[0], \
               self.target_size[0], \
               self.target_size[1], \
               input_shape[-1]
