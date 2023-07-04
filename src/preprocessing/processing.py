import numpy as np
import numpy.typing as npt
from typing import Any
from omegaconf import DictConfig
from typing import NamedTuple
from sklearn.preprocessing import OneHotEncoder
from preprocessing.loading import load_


class TrainValTest(NamedTuple):
    train: npt.NDArray[Any]
    val: npt.NDArray[Any]
    test: npt.NDArray[Any]


class Dataset(NamedTuple):
    x_picture: TrainValTest
    x_feature: TrainValTest
    y: TrainValTest


def _scale(x_picture: npt.NDArray[np.uint8],
           x_feature: npt.NDArray[np.uint8]) -> Any:
    """Scale
    
    `x_picture` is scaled by a factor of 255.
    
    `x_feature` is one-hot-encoded.
    
    Parameters
    ----------
    `x_picture: npt.NDArray[np.uint8]`
        Picture data.
    
    `x_feature: npt.NDArray[np.uint]`
        Feature data.
    
    Returns
    -------
    `npt.NDArray[np.float32], npt.NDArray[np.float64]`
        Scaled `x_picture` and `x_feature`, respectively.
    """
    x_picture: npt.NDArray[np.float32] = np.divide(x_picture,
                                                   255,
                                                   dtype=np.float32)

    x_feature_scaled: npt.NDArray[np.uint8] = (
        OneHotEncoder()
        .fit_transform(x_feature)
    )
    
    return x_picture, x_feature_scaled


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


def _train_val_test_split(x_picture: npt.NDArray[np.uint8],
                          x_feature: npt.NDArray[np.uint8],
                          y: npt.NDArray[np.uint8],
                          config: DictConfig) -> Dataset:
    """Train, validation and test split
    
    Split data among train, validation and test sets.
    
    Parameters
    ----------
    `x_picture: npt.NDArray[np.uint8]`
        Picture data.
    
    `x_feature: npt.NDArray[np.uint8]`
        Feature data.
    
    `y: npt.NDArray[np.uint8]`
        Labels.
    
    `config: DictConfig`
        Run configs.
    
    Returns
    -------
    `Dataset`
        Train, validation and test sets for pictures, features and labels.
    """
    m: int = y.shape[0]
    
    x_picture, x_feature = _scale(x_picture, x_feature)
    
    # Shuffle indexes
    indexes: npt.NDArray[np.int_] = np.array(list(range(m)))
    np.random.shuffle(indexes)
    
    # Set train and val sizes
    train_size: int = int(m * (1 - config.val_size - config.test_size))
    val_size: int = int(m * config.val_size)
    
    # Set train, val and test indices
    train_indices: npt.NDArray[np.int_] = indexes[:train_size]
    val_indices: npt.NDArray[np.int_] = indexes[train_size:val_size]
    test_indices: npt.NDArray[np.int_] = indexes[val_size:]
    
    # Make train, val and test sets for x_picture, x_feature and y
    x_picture_sets: TrainValTest = _split(x=x_picture,
                                          train_indices=train_indices,
                                          val_indices=val_indices,
                                          test_indices=test_indices)
    x_feature_sets: TrainValTest = _split(x=x_feature,
                                          train_indices=train_indices,
                                          val_indices=val_indices,
                                          test_indices=test_indices)
    y_sets: TrainValTest = _split(x=y,
                                  train_indices=train_indices,
                                  val_indices=val_indices,
                                  test_indices=test_indices)
    
    dataset: Dataset = Dataset(x_feature=x_feature_sets,
                               x_picture=x_picture_sets,
                               y=y_sets)
    
    return dataset


def preprocess(project_config: DictConfig,
               run_config: DictConfig) -> Dataset:
    x_picture, x_feature, y = load_(config=project_config)
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
    dataset: Dataset = _train_val_test_split(x_picture=x_picture,
                                             x_feature=x_feature,
                                             y=y,
                                             config=run_config)
    
    return(dataset)
