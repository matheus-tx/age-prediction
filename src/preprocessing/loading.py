import os
import re

from keras.utils import image_dataset_from_directory, img_to_array
from omegaconf import DictConfig
from PIL import Image


def _make_labels(set_: str, path: str, what: str) -> list[int]:
    filenames: str = list(os.walk(f'{path}/{set_}'))[0][2]
    filenames.sort()

    match what:
        case 'age':
            label_index: int = 1
        case 'gender':
            label_index = 2
        case 'race':
            label_index = 3

    labels: list[int] = [float(re.findall(r'\d+', filename)[label_index])
                         for filename in filenames]

    return labels


def make_generator(project_config: DictConfig,
                   run_config: DictConfig,
                   set_: str,
                   what: str):
    labels = _make_labels(set_=set_, path=project_config.data.path, what=what)

    dataset = image_dataset_from_directory(
        directory=f'{project_config.data.path}/{set_}',
        labels=labels,
        batch_size=run_config.batch_size,
        image_size=(run_config.image_size, run_config.image_size)
    )

    return dataset


def load_image(path: str, size: int):
    image = Image.open(path)
    image = image.resize(size=size)
    image_tensor = img_to_array(image)

    return image_tensor
