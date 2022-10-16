from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
from matplotlib.image import imread, imsave


def load_images() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = imread("data/x_train.tif")
    y_train = imread("data/y_train.gif").astype(bool)
    x_test = imread("data/x_test.tif")

    return x_train, y_train, x_test


def load_other_tests() -> List[np.ndarray]:
    return [imread(p.as_posix()) for p in Path("data/tests/").glob("*.tif")]


def save_images(mapping: Dict[str, np.ndarray]) -> None:
    for fn, im in mapping.items():
        imsave(fn, im, cmap='gray')
