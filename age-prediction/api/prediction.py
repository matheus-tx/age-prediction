import os

from api.poisson_model import make_poisson_model
from keras.models import Model, load_model
from keras.utils import get_file, image_dataset_from_directory
from scipy.stats import nbinom, poisson

from preprocessing import load_image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ALLOWED_MODEL_TYPES: list[str] = ['regular', 'poisson', 'negative_binomial']
BASE_WEIGHTS_PATH = (
    'https://github.com/matheus-tx/age-prediction/releases/download/release/'
)
IMAGE_SIZE = 200


def predict(images: str, model_type: str, coverage: float = 0):
    # Check if `model_type` is valid
    if model_type not in ALLOWED_MODEL_TYPES:
        msg: str = (
            f'`"{model_type}"` is not a valid value for `model_type`. It must be either '
            '`"regular"`, `"poisson"` or `"negative_binomial"`.'
        )
        raise ValueError(msg)

    # Load model
    file_name = f'age_predictor_{model_type}.keras.zip'
    model_path: str = get_file(fname=file_name,
                               origin=BASE_WEIGHTS_PATH + file_name,
                               extract=True,
                               cache_subdir='models/age-predictor')
    if model_type == 'poisson':
        model: Model = make_poisson_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    else:
        model: Model = load_model(model_path.replace('.zip', ''))

    # Load images
    if os.path.isdir(images):
        X = image_dataset_from_directory(directory=images,
                                         labels=None,
                                         image_size=(IMAGE_SIZE, IMAGE_SIZE))
    else:
        X = load_image(path=images, size=IMAGE_SIZE)

    # Predict age
    if coverage == 0:
        y_pred = model.predict(X)
    else:
        if model_type == 'regular':
            msg = (
                'It is not possible to predict intervals when parameter `model_type` is '
                '`"regular"`. Try setting `coverage` to `0` or setting `model` to either '
                '`"poisson"` or `"negative_binomial"`.'
            )
            raise ValueError(msg)
        y_params = model(X)
        if model == 'poisson':
            y_pred = poisson.interval(coverage, mu=y_params)
        else:
            pass
        # TODO: Add negative binomial distribution estimation.

    return y_pred
