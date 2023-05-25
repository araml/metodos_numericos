import numpy as np
import matplotlib.pyplot as plt
import deflate as d
from data_paths import *
from pathlib import Path
from random import choices
from string import ascii_letters

def read_images(path_to_images, use_smaller_images, scale_down_factor=1) -> np.array:
    paths = []
    images = []

    for path in sorted(list(path_to_images.rglob('*/*.pgm'))):
        paths.append(path)
        image = (plt.imread(path))
        if use_smaller_images:
            image = image[::scale_down_factor,::scale_down_factor] / 255
        images.append(image)

    return np.array(images)

# I don't like this function being here but for now it'll do
def get_eigenvalues_and_eigenvectors(M: np.array,
                                     k: int = 100,
                                     iters = 10,
                                     tolerance = 1e-17,
                                     filename = 'amogus') -> (np.array, np.array):
    e, v = d.deflate(M, np.ones(M.shape[0]), iters, k, tolerance)
    return np.array(e), np.array(v).T # return vectors as columns
