import os
import random
import re
from glob import glob

from omegaconf import DictConfig
from PIL import Image


def _clear_train_test_val(directory: str) -> None:
    sets = ['train', 'test', 'val']
    for set_ in sets:
        print(f'Clearing {set_} set...')
        files: list[str] = os.listdir(f'{directory}/{set_}')
        for file in files:
            os.remove(f'{directory}/{set_}/{file}')
    print('All previous files cleared!')


def split_train_val_test(project_config: DictConfig,
                         run_config: DictConfig,
                         seed: int | None = None):
    # Clear previous train, val and test sets before running
    _clear_train_test_val(project_config.data.path)

    # Randomly suffle images
    image_files: list[str] = glob(f'{project_config.data.path}/*.jpg')
    if seed is not None:
        random.Random(seed).shuffle(image_files)
    else:
        random.shuffle(image_files)

    n_images: int = len(image_files)
    val_frac: float = run_config.val_size
    test_frac: float = run_config.test_size

    train_size: int = int(n_images * (1 - val_frac - test_frac))
    val_size: int = train_size + int(n_images * val_frac)

    i = 0
    for filename in image_files:
        image = Image.open(fp=filename)

        age, gender, race = [
            int(label) for label in re.findall(r'\d+', filename)
        ][:3]

        if i < train_size:
            if not os.path.exists(project_config.data.train):
                os.mkdir(project_config.data.train)
            image.save(f'{project_config.data.train}/{i}_{age}_{gender}_{race}.jpg')
        elif i < val_size:
            if not os.path.exists(project_config.data.val):
                os.mkdir(project_config.data.val)
            image.save(f'{project_config.data.val}/{i}_{age}_{gender}_{race}.jpg')
        else:
            if not os.path.exists(project_config.data.test):
                os.mkdir(project_config.data.test)
            image.save(f'{project_config.data.test}/{i}_{age}_{gender}_{race}.jpg')

        i += 1
