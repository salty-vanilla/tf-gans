import os
import sys
import shutil
import yaml
sys.path.append(os.getcwd())
sys.path.append('../../')
from solver import Solver
from datasets.image_sampler import ImageSampler
from datasets.noise_sampler import NoiseSampler
from generator import Generator
from discriminator import Discriminator


def main():
    yml_path = sys.argv[1]
    with open(yml_path) as f:
        config = yaml.load(f)
    os.makedirs(config['logdir'], exist_ok=True)
    shutil.copy(yml_path, os.path.join(config['logdir'], 'config.yml'))

    image_sampler = ImageSampler(target_size=(4, 4),
                                 color_mode='rgb',
                                 is_training=True)
    noise_sampler = NoiseSampler('normal')

    generator = Generator(**config['generator_params'])
    discriminator = Discriminator(**config['discriminator_params'])

    solver = Solver(generator,
                    discriminator,
                    **config['solver_params'],
                    logdir=config['logdir'])

    solver.fit_generator(image_sampler.flow_from_tfrecord(config['records'],
                                                          shuffle=True),
                         noise_sampler,
                         **config['fit_params'])


if __name__ == '__main__':
    main()
