import deflate as d
import matplotlib.pyplot as plt
import numpy as np
from image_paths import * # TODO: rename image_paths since it also contains matrix path now
from pathlib import Path
from random import choices
from string import ascii_letters

USE_SMALLER_IMAGES = True # TODO: make this a CLI script option


def save_matrix_for_deflation(M: np.array, iters=10, tolerance=1e-17, filename=None) -> str:
    if not filename:
        filename = ''.join(choices(ascii_letters, k=12))
    file_path = Path(matrices_path, filename)
    dimension = M.shape[0]
    with open(file_path, 'w') as f:
        f.write("%d\n" % dimension)
        for i in range(dimension):
            for j in range(dimension):
                f.write("%.20f\n" % M[i,j])
        f.write("%d\n" % iters)
        f.write("%.20f" % tolerance)
    
    return str(file_path)

def read_images(path_to_images):
    paths = []
    images = []

    for path in sorted(list(path_to_images.rglob('*/*.pgm'))):
        paths.append(path)
        images.append(plt.imread(path))

    return images

def flatten_images(images: np.array) -> np.array:
    square_images = np.stack(images)
    flattened_images = square_images.reshape(square_images.shape[0], square_images[0].size)
    if USE_SMALLER_IMAGES:
        flattened_images = np.array([img[:20] for img in flattened_images])
    return flattened_images

def get_covariance_matrix(flattened_images: np.array) -> np.array:
    centred_images = flattened_images - np.mean(flattened_images, axis=0) # subtract average from each
    covariance = centred_images.T @ centred_images
    n = covariance.shape[0]
    covariance = covariance / (n-1)
    return covariance


images = read_images(Path(faces_path))
flattened_images = flatten_images(images)
covariance = get_covariance_matrix(flattened_images)
filename = save_matrix_for_deflation(covariance)

e, v = d.deflate(filename, np.ones(covariance.shape[0]), 2)
print(e,v)