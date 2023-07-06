from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig
from sklearn.preprocessing import OneHotEncoder

from preprocessing.loading import load_


class TrainValTest(NamedTuple):
    train: npt.NDArray[Any]
    val: npt.NDArray[Any]
    test: npt.NDArray[Any]


class Dataset(NamedTuple):
    x: TrainValTest
    y_age: TrainValTest
    y_gender: TrainValTest
    y_race: TrainValTest


def _scale(x: npt.NDArray[np.float32],
           y: npt.NDArray[np.float32]) -> Any:
    """Scale

    `x_picture` is scaled by a factor of 255.

    `x_feature` is one-hot-encoded.

    Parameters
    ----------
    `x_picture: npt.NDArray[np.float32]`
        Picture data.

    `y: npt.NDArray[np.uint]`
        Labels.

    Returns
    -------
    `npt.NDArray[np.float32], npt.NDArray[np.float64]`
        Scaled `x_picture` and `x_feature`, respectively.
    """
    x: npt.NDArray[np.float32] = np.divide(x,
                                           255,
                                           dtype=np.float32)

    y_age: npt.NDArray[np.float32] = y[:, 0].reshape(-1, 1) / 255
    y_gender: npt.NDArray[np.float32] = y[:, 1].reshape(-1, 1)
    y_race: npt.NDArray[np.float32] = \
        OneHotEncoder().fit_transform(y[:, 2].reshape(-1, 1))

    return x, y_age, y_gender, y_race


def _split(
    x: npt.NDArray[Any],
    train_indices: npt.NDArray[np.int_],
    val_indices: npt.NDArray[np.int_],
    test_indices: npt.NDArray[np.int_]
) -> TrainValTest:
    """Split

    Split among train, val and test sets.

    Parameters
    ----------
    `x: npt.NDArray[Any]`
        Array to be split.

    `train_indices: npt.NDArray[np.int_]`
        Indices for training set.

    `val_indices: npt.NDArray[npt.int_]`
        Indices for validation set.

    `test_indices: npt.NDArray[npt.int_]`
        Indices for test set.

    Returns
    -------
    `TrainValTest`
        Train, validation and test sets.
    """
    x_train: npt.NDArray[Any] = x[train_indices, :]
    x_val: npt.NDArray[Any] = x[val_indices, :]
    x_test: npt.NDArray[Any] = x[test_indices, :]

    train_val_test = TrainValTest(train=x_train,
                                  val=x_val,
                                  test=x_test)

    return train_val_test


def _train_val_test_split(x: npt.NDArray[np.float32],
                          y: npt.NDArray[np.float32],
                          config: DictConfig) -> Dataset:
    """Train, validation and test split

    Split data among train, validation and test sets.

    Parameters
    ----------
    `x: npt.NDArray[np.float32]`
        Picture data.

    `y: npt.NDArray[np.float32]`
        Labels.

    `config: DictConfig`
        Run configs.

    Returns
    -------
    `Dataset`
        Train, validation and test sets for pictures, features and labels.
    """
    m: int = y.shape[0]

    x, y_age, y_gender, y_race = _scale(x, y)

    # Shuffle indexes
    indexes: npt.NDArray[np.int_] = np.array(list(range(m)))
    np.random.shuffle(indexes)

    # Set train and val sizes
    train_size: int = int(m * (1 - config.val_size - config.test_size))
    val_size: int = train_size + int(m * config.val_size)

    # Set train, val and test indices
    train_indices: npt.NDArray[np.int_] = indexes[:train_size]
    val_indices: npt.NDArray[np.int_] = indexes[train_size:val_size]
    test_indices: npt.NDArray[np.int_] = indexes[val_size:]

    # Make train, val and test sets for x_picture, x_feature and y
    x: TrainValTest = _split(x=x,
                             train_indices=train_indices,
                             val_indices=val_indices,
                             test_indices=test_indices)
    y_age_sets: TrainValTest = _split(x=y_age,
                                      train_indices=train_indices,
                                      val_indices=val_indices,
                                      test_indices=test_indices)
    y_gender_sets: TrainValTest = _split(x=y_gender,
                                         train_indices=train_indices,
                                         val_indices=val_indices,
                                         test_indices=test_indices)
    y_race_sets: TrainValTest = _split(x=y_race,
                                       train_indices=train_indices,
                                       val_indices=val_indices,
                                       test_indices=test_indices)

    dataset: Dataset = Dataset(x=x,
                               y_age=y_age_sets,
                               y_gender=y_gender_sets,
                               y_race=y_race_sets)

    return dataset


def preprocess(project_config: DictConfig,
               run_config: DictConfig) -> Dataset:
    """Preprocess

    Do all preprocessing. It includes loading data, scaling it and splitting
    it among train, validation and test sets.

    Parameters
    ----------
    `project_config: DictConfig`
        Project configs.

    `run_config: DictConfig`
        Run configs.

    Returns
    -------
    `Dataset`
        Scaled train, validation and test sets for data.
    """
    x, y = load_(config=project_config)

    dataset: Dataset = _train_val_test_split(x=x,
                                             y=y,
                                             config=run_config)

    return(dataset)
