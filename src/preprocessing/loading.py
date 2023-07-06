import re
from glob import glob
from typing import Any

import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig
from PIL import Image

EXPECTED_FEATURE_SHAPE: tuple[int] = (4, )
NEW_PICTURE_SHAPE: tuple[int] = (100, 100)


def load_(
    config: DictConfig
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
