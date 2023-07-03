from PIL import Image
from omegaconf import DictConfig
import numpy.typing as npt

def load_images(config: DictConfig) -> npt.NDArray