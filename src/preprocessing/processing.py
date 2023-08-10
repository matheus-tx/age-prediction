import os
import re
from glob import glob

from omegaconf import DictConfig
from PIL import Image


def split_train_val_test(project_config: DictConfig,
                         run_config: DictConfig):
    # TODO: delete all files in training, validation and test paths before splitting
    # dataset.
    image_files: list[str] = glob(f'{project_config.data.path}/*.jpg')
    n_images: int = len(image_files)
    val_frac: float = run_config.val_size
    test_frac: float = run_config.test_size

    train_size: int = int(n_images * (1 - val_frac - test_frac))
    val_size: int = train_size + int(n_images * val_frac)

    i = 0
    for filename in image_files:
        image = Image.open(fp=filename)

        age, gender, race = (
            int(label) for label in re.findall(r'\d+', filename)
        )[:2]

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
