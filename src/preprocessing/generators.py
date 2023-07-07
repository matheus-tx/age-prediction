import os

import numpy as np
import numpy.typing as npt
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from omegaconf import DictConfig
from PIL import Image

from preprocessing.processing import Dataset, preprocess

TARGET_SIZE: tuple = (128, 128)


# def _save_(project_config: DictConfig,
#            run_config: DictConfig) -> None:
#     os.mkdir(project_config.data[set])

#     n: int = dataset.x.__getattribute__(set).shape[0]
#     for i in range(n):
#         x: npt.NDArray[np.float32] = dataset.x.__getattribute__(set)[i]
#         age: int = \
#             int(dataset.y_age.__getattribute__(set).flatten()[i])
#         gender: int = \
#             int(dataset.y_gender.__getattribute__(set).flatten()[i])
#         race: npt.NDArray[np.float32] = \
#             int(dataset.y_race.__getattribute__(set).flatten()[i])

#         age_folder: str = f'{project_config.data[set]}/{age}'
#         if not os.path.exists(age_folder):
#             os.mkdir(age_folder)

#         image: Image = Image.fromarray(x, mode='RGB')
#         image.save(f'{age_folder}/{i}_{age}_{gender}_{race}.jpg')


# def split_data(project_config: DictConfig,
#                run_config: DictConfig) -> None:
#     """Split data

#     Split image data among train, validation and test sets. Each set belongs
#     to one folder.

#     Arguments
#     ---------
#     `config: DictConfig`
#         Project configs.
#     """
#     if os.path.exists(project_config.data.train):
#         pass
#     else:
#         dataset: Dataset = preprocess(project_config=project_config,
#                                       run_config=run_config)

#         _save_(dataset=dataset, set='train', project_config=project_config)
#         _save_(dataset=dataset, set='val', project_config=project_config)
#         _save_(dataset=dataset, set='test', project_config=project_config)


def create_flow(df: pd.DataFrame,
                project_config: DictConfig,
                which_set: str,
                batch_size: int = 64) -> ImageDataGenerator:
    """Create flow

    Create image flow from directory.

    Arguments
    ---------
    `set: str`
        Which set (among `train`, `val` and `test`) to create flow from.

    `batch_size: int = 32`
        Batch size.

    Returns
    -------
    `ImageDataGenerator`
        Image flow.
    """
    if which_set == 'train':
        image_generator: ImageDataGenerator = ImageDataGenerator(
            rescale=1 / 255,
            rotation_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=(0.25, 0.5),
            channel_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
        )
    else:
        image_generator = ImageDataGenerator(rescale=1 / 255)

    gen = image_generator.flow_from_dataframe(
        dataframe=df,
        directory=project_config.data.path,
        y_col='age',
        target_size=TARGET_SIZE,
        class_mode='raw',
        batch_size=batch_size
    )

    return gen
