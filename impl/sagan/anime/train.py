import os
import sys
import shutil
import yaml
sys.path.append(os.getcwd())
sys.path.append('../../')
from solver import Solver
from datasets.image_sampler import ImageSampler
from datasets.noise_sampler import NoiseSampler
# from models.generator import SRResnetGenerator as Generator
# from models.discriminator import SRResnetDiscriminator as Discriminator
from anime.generator import DeepResidualAnimeGenerator128 as Generator
from anime.discriminator import ResidualAnimeDiscriminator128 as Discriminator


def main():
    yml_path = sys.argv[1]
    with open(yml_path) as f:
        config = yaml.load(f)
    os.makedirs(config['logdir'], exist_ok=True)
    shutil.copy(yml_path, os.path.join(config['logdir'], 'config.yml'))

    image_sampler = ImageSampler(target_size=(128, 128),
                                 color_mode='rgb',
                                 is_training=True)
    noise_sampler = NoiseSampler('uniform')

    generator = Generator(**config['generator_params'])
    discriminator = Discriminator(**config['discriminator_params'])
    solver = Solver(generator,
                    discriminator,
                    **config['solver_params'],
                    logdir=config['logdir'])

    solver.fit_generator(image_sampler.flow_from_tfrecord(config['records'],
                                                          batch_size=config['batch_size'],
                                                          shuffle=True,
                                                          is_random_flip=True),
                         noise_sampler,
                         **config['fit_params'])


if __name__ == '__main__':
    main()