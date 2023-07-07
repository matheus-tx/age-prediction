import os
import random
import re
from glob import glob
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from keras.utils import image_dataset_from_directory
from omegaconf import DictConfig
from PIL import Image

EXPECTED_FEATURE_SHAPE: tuple[int] = (4, )
NEW_PICTURE_SHAPE: tuple[int] = (128, 128)


def load_(
    config: DictConfig,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """"Load

    Load all image and feature data. It does so because features are encoded
    in image names.

    Parameters
    ----------
    `config: DictConfig`
        Project configs.

    Returns
    `npt.NDArray[np.unit8]`
    """
    image_files = glob(f'{config.data.path}/*.jpg')

    images: list[npt.NDArray[np.float32]] = []
    y_list: list[npt.NDArray[np.float32]] = []

    for file in image_files:
        image = Image.open(fp=file)
        image = image.resize(size=NEW_PICTURE_SHAPE)
        images.append(np.array(image, dtype=np.float32))
        image.close()

        y: npt.NDArray[np.float32] = np.array(
            [int(feature) for feature in re.findall(r'\d+', file)],
            dtype=np.float32
        )

        if y.shape != (4, ):
            y = np.insert(y, -1, 4)
        y = y[:-1]

        y_list.append(y)

    image_array: npt.NDArray[np.float32] = np.stack(images)
    y_array: npt.NDArray[np.float32] = np.stack(y_list)

    return image_array, y_array


def make_dataframe(project_config: DictConfig,
                   run_config: DictConfig) -> pd.DataFrame:
    """Make dataframe

    Make dataframe with input image names and its labels.

    Arguments
    ---------
    `config: DictConfig`
        Project configs.

    Returns
    -------
    `pd.DataFrame`
        Input data.
    """
    image_filenames: list[str] = glob(f'{project_config.data.path}/*.jpg')

    image_list: list[list[Any]] = []
    for filename in image_filenames:
        row: list[Any] = [filename.replace('data/', '')]
        y: list[int] = [
            int(label) for label in re.findall(r'\d+', filename)
        ][:-1]

        if len(y) < 3:
            y.append(4)
        row.extend(y)

        image_list.append(row)

    image_df: pd.DataFrame = pd.DataFrame(
        image_list,
        columns=['filename', 'age', 'gender', 'race']
    )

    image_df = image_df.sample(frac=1)
    m: int = image_df.shape[0]
    train_size: int = int(
        m * (1 - run_config.val_size - run_config.test_size)
    )
    val_size: int = int(m * (1 - run_config.test_size))

    train: pd.DataFrame = image_df.iloc[:train_size].reset_index(drop=True)
    val: pd.DataFrame = \
        image_df.iloc[train_size:val_size].reset_index(drop=True)
    test = pd.DataFrame = image_df.iloc[val_size:].reset_index(drop=True)

    return train, val, test


def split(project_config: DictConfig,
          run_config: DictConfig):
    image_files: list[str] = glob(f'{project_config.data.path}/*.jpg')
    n_images: int = len(image_files)
    val_frac: float = run_config.val_size
    test_frac: float = run_config.test_size

    train_size: int = int(n_images * (1 - val_frac - test_frac))
    val_size: int = train_size + int(n_images * val_frac)

    i = 0
    for filename in image_files:
        image = Image.open(fp=filename)

        age: int = [int(label) for label in re.findall(r'\d+', filename)][0]

        if i < train_size:
            if not os.path.exists(project_config.data.train):
                os.mkdir(project_config.data.train)
            image.save(f'{project_config.data.train}/{i}_{age}.jpg')
        elif i < val_size:
            if not os.path.exists(project_config.data.val):
                os.mkdir(project_config.data.val)
            image.save(f'{project_config.data.val}/{i}_{age}.jpg')
        else:
            if not os.path.exists(project_config.data.test):
                os.mkdir(project_config.data.test)
            image.save(f'{project_config.data.test}/{i}_{age}.jpg')

        i += 1


def load_images(directory: str):
    y: list[int] = [
        int(re.findall(r'\d+', filename)[-1])
        for filename
        in list(os.walk(directory))[-1][-1]
    ]

    images = image_dataset_from_directory(
        directory=directory,
        labels=y,
        batch_size=32,
        shuffle=False
    )

    return images
