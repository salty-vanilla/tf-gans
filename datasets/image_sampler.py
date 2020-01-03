import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import Iterator
import os
import numpy as np
from PIL import Image, ImageOps


class ImageSampler:
    def __init__(self, target_size=None,
                 color_mode='rgb',
                 normalize_mode='tanh',
                 is_training=True):
        self.target_size = target_size
        self.color_mode = color_mode
        self.normalize_mode = normalize_mode
        self.is_training = is_training

    def flow(self, x,
             y=None,
             batch_size=32,
             shuffle=True,
             seed=None,
             is_random_flip=False):
        return ArrayIterator(x,
                             y,
                             target_size=self.target_size,
                             color_mode=self.color_mode,
                             batch_size=batch_size,
                             normalize_mode=self.normalize_mode,
                             shuffle=shuffle,
                             seed=seed,
                             is_training=self.is_training,
                             is_random_flip=is_random_flip)

    def flow_from_directory(self, image_dir,
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            nb_sample=None,
                            is_random_flip=False):
        image_paths = np.array([path for path in get_image_paths(image_dir)])
        if nb_sample is not None:
            image_paths = image_paths[:nb_sample]

        return DirectoryIterator(paths=image_paths,
                                 target_size=self.target_size,
                                 color_mode=self.color_mode,
                                 batch_size=batch_size,
                                 normalize_mode=self.normalize_mode,
                                 shuffle=shuffle,
                                 seed=seed,
                                 is_training=self.is_training,
                                 is_random_flip=is_random_flip)

    def flow_from_path(self, paths,
                       batch_size=32,
                       shuffle=True,
                       seed=None,
                       nb_sample=None,
                       is_random_flip=False):
        if nb_sample is not None:
            image_paths = np.array(paths[:nb_sample])
        else:
            image_paths = np.array(paths)
        return DirectoryIterator(paths=image_paths,
                                 target_size=self.target_size,
                                 color_mode=self.color_mode,
                                 batch_size=batch_size,
                                 normalize_mode=self.normalize_mode,
                                 shuffle=shuffle,
                                 seed=seed,
                                 is_training=self.is_training,
                                 is_random_flip=is_random_flip)

    def flow_from_tfrecord(self, file_paths,
                           batch_size=32,
                           shuffle=True,
                           seed=None,
                           is_random_flip=False):
        return TFRecordIterator(file_paths,
                                target_size=self.target_size,
                                color_mode=self.color_mode,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                seed=seed,
                                is_training=self.is_training,
                                is_random_flip=is_random_flip)


class DirectoryIterator(Iterator):
    def __init__(self, paths,
                 target_size,
                 color_mode,
                 batch_size,
                 normalize_mode,
                 shuffle,
                 seed,
                 is_training,
                 is_random_flip):
        self.paths = paths
        self.target_size = target_size
        self.color_mode = color_mode
        self.nb_sample = len(self.paths)
        self.batch_size = batch_size
        self.normalize_mode = normalize_mode
        super().__init__(self.nb_sample, batch_size, shuffle, seed)
        self.current_paths = None
        self.is_training = is_training
        self.is_random_flip = is_random_flip

    def __call__(self, *args, **kwargs):
        if self.is_training:
            return self.flow_on_training()
        else:
            return self.flow_on_test()

    def flow_on_training(self):
        with self.lock:
            index_array = next(self.index_generator)
        image_path_batch = self.paths[index_array]
        image_batch = np.array([load_image(path,
                                           self.target_size,
                                           self.color_mode,
                                           self.is_random_flip)
                                for path in image_path_batch])
        self.current_paths = image_path_batch
        return image_batch

    def flow_on_test(self):
        indexes = np.arange(self.nb_sample)
        if self.shuffle:
            np.random.shuffle(indexes)

        steps = self.nb_sample // self.batch_size
        if self.nb_sample % self.batch_size != 0:
            steps += 1
        for i in range(steps):
            index_array = indexes[i * self.batch_size: (i + 1) * self.batch_size]
            image_path_batch = self.paths[index_array]
            image_batch = np.array([load_image(path,
                                               self.target_size,
                                               self.color_mode,
                                               self.is_random_flip)
                                    for path in image_path_batch])

            self.current_paths = image_path_batch
            yield image_batch

    def data_to_image(self, x):
        x = np.array(x)
        if x.shape[-1] == 1:
            x = x.reshape(x.shape[:-1])
        return denormalize(x, self.normalize_mode)

    def __len__(self):
        return self.nb_sample


class ArrayIterator(Iterator):
    def __init__(self, x,
                 y,
                 target_size,
                 color_mode,
                 batch_size,
                 normalize_mode,
                 shuffle,
                 seed,
                 is_training,
                 is_random_flip):
        self.x = x
        self.y = y
        self.target_size = target_size
        self.color_mode = color_mode
        self.nb_sample = len(self.x)
        self.batch_size = batch_size
        self.normalize_mode = normalize_mode
        super().__init__(self.nb_sample, batch_size, shuffle, seed)
        self.is_training = is_training
        self.is_random_flip = is_random_flip

        if len(self.x.shape) == 4:
            if x.shape[3] == 1:
                self.x = self.x.reshape(self.x.shape[:3])

    def __call__(self, *args, **kwargs):
        if self.is_training:
            return self.flow_on_training()
        else:
            return self.flow_on_test()

    def flow_on_training(self):
        with self.lock:
            index_array = next(self.index_generator)
        image_batch = np.array([preprocessing(Image.fromarray(x),
                                              color_mode=self.color_mode,
                                              target_size=self.target_size,
                                              is_random_flip=self.is_random_flip)
                                for x in self.x[index_array]])
        if self.y is not None:
            label_batch = self.y[index_array]
            return image_batch, label_batch
        else:
            return image_batch

    def flow_on_test(self):
        indexes = np.arange(self.nb_sample)
        if self.shuffle:
            np.random.shuffle(indexes)

        steps = self.nb_sample // self.batch_size
        if self.nb_sample % self.batch_size != 0:
            steps += 1
        for i in range(steps):
            index_array = indexes[i * self.batch_size: (i + 1) * self.batch_size]
            image_batch = np.array([preprocessing(Image.fromarray(x),
                                                  color_mode=self.color_mode,
                                                  target_size=self.target_size,
                                                  is_random_flip=self.is_random_flip)
                                    for x in self.x[index_array]])
            if self.y is not None:
                label_batch = self.y[index_array]
                yield image_batch, label_batch
            else:
                yield image_batch

    def data_to_image(self, x):
        x = np.array(x)
        return denormalize(x, self.normalize_mode)

    def __len__(self):
        return self.nb_sample


def preprocessing(x,
                  target_size=None,
                  color_mode='rgb',
                  is_random_flip=False):
    assert color_mode in ['grayscale', 'gray', 'rgb']
    if color_mode in ['grayscale', 'gray']:
        image = x.convert('L')
    else:
        image = x

    if is_random_flip and np.random.uniform(0., 1.) > 0.5:
        image = ImageOps.mirror(image)

    if target_size is not None and target_size != image.size:
        image = image.resize(target_size, Image.BILINEAR)

    image_array = np.asarray(image)

    image_array = normalize(image_array)

    if image_array.ndim == 2:
        image_array = np.expand_dims(image_array, axis=-1)
    return image_array


def load_image(path,
               target_size=None,
               color_mode='rgb',
               is_random_flip=False):
    image = Image.open(path)
    try:
        return preprocessing(image,
                             target_size,
                             color_mode,
                             is_random_flip)
    except:
        print(path)
        exit()


def normalize(x, mode='tanh'):
    if mode == 'tanh':
        return (x.astype('float32') / 255 - 0.5) / 0.5
    elif mode == 'sigmoid':
        return x.astype('float32') / 255
    else:
        raise NotImplementedError


def denormalize(x, mode='tanh'):
    if mode == 'tanh':
        return ((x + 1.) / 2 * 255).astype('uint8')
    elif mode == 'sigmoid':
        return (x * 255).astype('uint8')
    else:
        raise NotImplementedError


def get_image_paths(src_dir):
    def get_all_paths():
        for root, dirs, files in os.walk(src_dir):
            yield root
            for file in files:
                yield os.path.join(root, file)

    def is_image(path):
        valid_exts = ['png', 'jpg', 'jpeg', 'bmp', 'PNG', 'JPG', 'JPEG', 'BMP']
        _, ext = os.path.splitext(path)
        ext = ext[1:]
        if ext in valid_exts:
            return True
        else:
            return False

    return [path for path in get_all_paths() if is_image(path)]


class TFRecordIterator:
    def __init__(self, file_paths,
                 map_fn=None,
                 compression_type='GZIP',
                 target_size=None,
                 color_mode='rgb',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 is_training=True,
                 is_random_flip=False):
        self.file_paths = file_paths
        self.nb_sample = 0
        if map_fn is not None:
            self.map_dataset = map_fn
        self.compression_type = compression_type
        self._target_size = target_size
        self.color_mode = color_mode
        self._batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.is_training = is_training
        self.is_random_flip = is_random_flip
        self._build_dataset()

    def __next__(self):
        return next(self.iterator)

    def __call__(self):
        return next(self.iterator)

    def __iter__(self):
        return self.iterator

    @staticmethod
    def data_to_image(x: tf.Tensor):
        if x.get_shape().as_list()[-1] == 1:
            x = x.reshape(x.get_shape().as_list()[:-1])
        x = np.array(x)
        return denormalize(x)

    def map_dataset(self, serialized):
        features = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'channel': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string),
        }
        features = tf.io.parse_single_example(serialized, features)
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        channel = tf.cast(features['channel'], tf.int32)
        images = tf.io.decode_raw(features['image'], tf.uint8)
        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, (height, width, channel))
        images = tf.image.resize(images, self._target_size)
        if self.is_random_flip:
            images = tf.image.random_flip_left_right(images)
        return (images / 255 - 0.5) / 0.5

    def _build_dataset(self):
        self.dataset = tf.data.TFRecordDataset(self.file_paths, compression_type=self.compression_type)
        self.dataset = self.dataset.repeat(1 if not self.is_training else -1)
        self.dataset = self.dataset.map(self.map_dataset)
        if self.shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=256, seed=self.seed)
        self.dataset = self.dataset.batch(self._batch_size)
        self.iterator = self.dataset.__iter__()

    def __len__(self):
        if self.nb_sample:
            return self.nb_sample
        else:
            print('\nComputing the number of records now.'),
            print('It takes a long time only for the 1st time\n')
            options = tf.io.TFRecordOptions(self.compression_type)
            for path in self.file_paths:
                self.nb_sample += sum(1 for _ in tf.data.TFRecordDataset(path, compression_type=self.compression_type).batch(1))
        return self.nb_sample

    @property
    def target_size(self):
        return self._target_size

    @target_size.setter
    def target_size(self, target_size):
        self._target_size = target_size
        self._build_dataset()

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size
        self._build_dataset()
