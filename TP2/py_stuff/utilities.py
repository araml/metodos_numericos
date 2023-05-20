import numpy as np
import matplotlib.pyplot as plt
import deflate as d
from image_paths import *
from pathlib import Path

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
                                     filename = 'amogus',
                                     get_all = False) -> (np.array, np.array):
    if get_all:
        number_of_eigenvalues = M.shape[0]
    else:
        number_of_eigenvalues = k

    print("\tSaving matrix...")
    filename = save_matrix_for_deflation(M, iters, tolerance, filename)
    print("\tSaved matrix to file {}".format(filename))

    print("\tDeflating...")
    e, v = d.deflate(filename, np.ones(M.shape[0]), number_of_eigenvalues)
    return np.array(e), np.array(v).T # return vectors as columns

# common functions

def save_matrix_for_deflation(M: np.array, iters=10, tolerance=1e-17, filename=None) -> str:
    if not filename:
        filename = ''.join(choices(ascii_letters, k=12))
    file_path = Path(matrices_path, filename)
    dimension = M.shape[0]
    with open(file_path, 'w') as f:
        f.write("%d\n" % dimension)
        for i in range(dimension):
            for j in range(dimension):
                f.write("%.7f\n" % M[i,j])
        f.write("%d\n" % iters)
        f.write("%.20f" % tolerance)
    
    return str(file_path)
