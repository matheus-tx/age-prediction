from PIL import Image
from omegaconf import DictConfig
import numpy as np
import numpy.typing as npt
from glob import glob
from typing import Any
import re

EXPECTED_FEATURE_SHAPE: tuple[int] = (4, )
NEW_PICTURE_SHAPE: tuple[int] = (100, 100)

def load_(
    config: DictConfig
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
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

    images: list[npt.NDArray[np.uint8]] = []
    feature_list: list[npt.NDArray[np.uint8]] = []
    for file in image_files:
        image = Image.open(fp=file)
        image = image.resize(size=NEW_PICTURE_SHAPE)
        images.append(np.array(image, dtype=np.uint8))
        image.close()
        
        features: npt.NDArray[np.uint8] = np.array(
            [int(feature) for feature in re.findall(r'\d+', file)],
            dtype=np.uint8
        )
        
        if features.shape != (4, ):
            features = np.insert(features, -1, 4)
        features = features[:-1]
            
        feature_list.append(features)
        
    image_array: npt.NDArray[np.uint8] = np.stack(images)
    feature_array: npt.NDArray[np.uint8] = np.stack(feature_list)
    y: npt.NDArray[np.uint8] = feature_array[:, 0].reshape(-1, 1)
    x: npt.NDArray[np.uint8] = feature_array[:, 1:]

    return image_array, x, y