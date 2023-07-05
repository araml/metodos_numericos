import numpy as np
import matplotlib.pyplot as plt
import deflate as d
import time
from data_paths import *


def read_images(path_to_images, scale_down_factor=1) -> np.array:
    paths = []
    images = []

    for path in sorted(list(path_to_images.rglob('*/*.pgm'))):
        paths.append(path)
        image = (plt.imread(path))
        image = image[::scale_down_factor,::scale_down_factor]
        images.append(image)

    return np.array(images)

# I don't like this function being here but for now it'll do
def get_eigenvalues_and_eigenvectors(M: np.array,
                                     k: int = 100,
                                     iters = 10,
                                     tolerance = 1e-17) -> (np.array, np.array):
    e, v = d.deflate(M, np.ones(M.shape[0]), iters, k, tolerance)
    return np.array(e), np.array(v).T # return vectors as columns

def flatten_images(images: np.array) -> np.array:
    square_images = np.stack(images)
    flattened_images = square_images.reshape(square_images.shape[0], 
                                                square_images[0].size)
    return flattened_images

def centre_images(images: np.array) -> np.array:
    # subtract average from each
    flattened_images = flatten_images(images)
    return flattened_images - np.mean(flattened_images, axis=0)

def measure_execution_time(function_to_measure, *args) -> float:
    start_time = time.time()
    function_to_measure(*args)
    end_time = time.time()
    return end_time-start_time