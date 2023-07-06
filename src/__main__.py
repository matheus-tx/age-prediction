import argparse

import hydra
from omegaconf import DictConfig

from modelling.model import PoissonAgePredictor
from preprocessing import Dataset, preprocess

if __name__ == '__main__':
    # Load configs
    hydra.initialize(config_path='../config')
    project_config: DictConfig = hydra.compose(config_name='main')

    # Run configs
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--val-size',
                        help='Validation set size',
                        type=float)
    parser.add_argument('-t',
                        '--test-size',
                        help='Test set size',
                        type=float)
    run_config: DictConfig = DictConfig(vars(parser.parse_args()))

    # Load data
    dataset: Dataset = preprocess(project_config=project_config,
                                  run_config=run_config)

    model = PoissonAgePredictor(data=dataset,
                                config=run_config)
    model.fit()

    pass
