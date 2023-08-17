import argparse

import hydra
from omegaconf import DictConfig

from modelling.model import AgePredictor
from preprocessing import load_image, make_generator, split_train_val_test

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
    parser.add_argument('-p',
                        '--image-path',
                        help='Path to image to predict',
                        type=str)
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

    # Train model
    model = AgePredictor(input_shape=train.element_spec[0].shape[1:])
    model.fit(train, val, run_config.n_epochs)

    # Test model
    test = make_generator(project_config=project_config,
                          run_config=run_config,
                          set_='test',
                          what='age')
    error = model.evaluate(test_data=test)
    print(f'Mean error of model on test set is {error:.2f} years.')

    # Predict
    image_to_predict = load_image(run_config.path, model.input_shape)
    prediction = model.predict(image_to_predict)
    print(f'Predicted age for image is {prediction} years.')

    pass
