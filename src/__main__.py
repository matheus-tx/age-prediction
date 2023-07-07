import argparse

import hydra
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory
from omegaconf import DictConfig

from modelling.model import PoissonAgePredictor
from preprocessing import Dataset, preprocess
from preprocessing.generators import create_flow, split_data
from preprocessing.loading import load_images, make_dataframe, split

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

    # split(project_config, run_config)

    # Load data
    # train_df, val_df, test_df = make_dataframe(
    #     project_config=project_config,
    #     run_config=run_config
    # )

    # split_data(project_config=project_config, run_config=run_config)

    # train_set = create_flow(df=train,
    #                         project_config=project_config,
    #                         which_set='train'age_dataset_from_directory(
    # directory=project_config.data.train,
    # labels='inferred',
    # label_mode='int',
    # image_size=(100, 100),
    # batch_size=64)
    # val_set = create_flow(df=val,
    #                       project_config=project_config,
    #                       which_set='val')
    # test_set = create_flow(df=test,
    #                        project_config=project_config,
    #                        which_set='test')
    # dataset: Dataset = preprocess(project_config=project_config,
    #                               run_config=run_config)

    train = load_images(project_config.data.train)

    model = PoissonAgePredictor(train_set=train,
                                # val_set=val_set,
                                # test_set=test_set,
                                # dataset=dataset,
                                config=run_config)
    model.fit()

    pass
