import os
import sys
import shutil
import yaml
sys.path.append(os.getcwd())
sys.path.append('../../')
from solver import Solver
from datasets.mnist import load_data
from datasets.noise_sampler import NoiseSampler
from models.generator import MNISTGenerator as Generator
from models.discriminator import MNISTDiscriminator as Discriminator


def main():
    yml_path = sys.argv[1]
    with open(yml_path) as f:
        config = yaml.load(f)
    os.makedirs(config['logdir'], exist_ok=True)
    shutil.copy(yml_path, os.path.join(config['logdir'], 'config.yml'))

    x = load_data()
    noise_sampler = NoiseSampler('normal')

    generator = Generator(**config['generator_params'])
    discriminator = Discriminator(**config['discriminator_params'])

    solver = Solver(generator,
                    discriminator,
                    **config['solver_params'],
                    logdir=config['logdir'])

    solver.fit(x,
               noise_sampler,
               **config['fit_params'])


if __name__ == '__main__':
    main()
