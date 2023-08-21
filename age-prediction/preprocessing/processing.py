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
    # Create training, validation and test sets
    data_not_split_yet: bool = False
    if not os.path.exists(project_config.data.train):
        os.mkdir(project_config.data.train)
        data_not_split_yet = True
    if not os.path.exists(project_config.data.val):
        os.mkdir(project_config.data.val)
    if not os.path.exists(project_config.data.test):
        os.mkdir(project_config.data.test)

    if data_not_split_yet or run_config.train_val_test_split:
        # Clear previous train, val and test sets before
        if not data_not_split_yet:
            _clear_train_test_val(project_config.data.path)
        else:
            print(
                'Data has not been split among training, validation and test sets.'
                'Splitting it now...'
            )

        # Randomly suffle images before splitting sets
        image_files: list[str] = glob(f'{project_config.data.path}/*.jpg')
        if seed is not None:
            random.Random(seed).shuffle(image_files)
        else:
            random.shuffle(image_files)

        # Compute set sizes
        n_images: int = len(image_files)
        val_frac: float = run_config.val_size
        test_frac: float = run_config.test_size
        train_size: int = int(n_images * (1 - val_frac - test_frac))
        val_size: int = train_size + int(n_images * val_frac)

        # Split and save images
        i = 0
        for filename in image_files:
            image = Image.open(fp=filename)

            age, gender, race = [
                int(label) for label in re.findall(r'\d+', filename)
            ][:3]

            if i < train_size:
                image.save(f'{project_config.data.train}/{i}_{age}_{gender}_{race}.jpg')
            elif i < val_size:
                image.save(f'{project_config.data.val}/{i}_{age}_{gender}_{race}.jpg')
            else:
                image.save(f'{project_config.data.test}/{i}_{age}_{gender}_{race}.jpg')

            i += 1
