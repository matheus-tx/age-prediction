import argparse

import hydra
from omegaconf import DictConfig

from modelling.model import AgePredictor
from preprocessing import make_generator, split_train_val_test

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
    parser.add_argument('-b',
                        '--batch-size',
                        help='Batch size',
                        type=int)
    parser.add_argument('-i',
                        '--image-size',
                        help='Image size',
                        type=int)
    parser.add_argument('-s',
                        '--train-val-test-split',
                        help='Split training, validation and test sets',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('-e',
                        '--n-epochs',
                        help='Number of epochs',
                        type=int)
    run_config: DictConfig = DictConfig(vars(parser.parse_args()))

    split_train_val_test(project_config=project_config,
                         run_config=run_config,
                         seed=1)

    train = make_generator(project_config=project_config,
                           run_config=run_config,
                           set_='train',
                           what='age')
    val = make_generator(project_config=project_config,
                         run_config=run_config,
                         set_='val',
                         what='age')

    model = AgePredictor(input_shape=train.element_spec[0].shape[1:])
    model.fit(train, val, run_config.n_epochs)

    pass

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

    # train = load_images(project_config.data.train)

    # model = PoissonAgePredictor(train_set=train,
    #                             # val_set=val_set,
    #                             # test_set=test_set,
    #                             # dataset=dataset,
    #                             config=run_config)
    # model.fit()

    pass
