import deflate as d
import matplotlib.pyplot as plt
import numpy as np
from image_paths import * # TODO: rename image_paths since it also contains matrix path now
from pathlib import Path
from random import choices
from string import ascii_letters

USE_SMALLER_IMAGES = True # TODO: make this a CLI script option
SCALE_DOWN_FACTOR = 2


def save_matrix_for_deflation(M: np.array, iters=10, tolerance=1e-17, filename=None) -> str:
    if not filename:
        filename = ''.join(choices(ascii_letters, k=12))
    file_path = Path(matrices_path, filename)
    dimension = M.shape[0]
    with open(file_path, 'w') as f:
        f.write("%d\n" % dimension)
        for i in range(dimension):
            for j in range(dimension):
                f.write("%.5f\n" % M[i,j])
        f.write("%d\n" % iters)
        f.write("%.20f" % tolerance)
    
    return str(file_path)


def read_images(path_to_images):
    paths = []
    images = []

    for path in sorted(list(path_to_images.rglob('*/*.pgm'))):
        paths.append(path)
        image = (plt.imread(path))
        if USE_SMALLER_IMAGES:
            image = image[::SCALE_DOWN_FACTOR,::SCALE_DOWN_FACTOR] / 255
        images.append(image)

    return images


def flatten_images(images: np.array) -> np.array:
    square_images = np.stack(images)
    flattened_images = square_images.reshape(square_images.shape[0], square_images[0].size)
    return flattened_images


def get_covariance_matrix(flattened_images: np.array) -> np.array:
    n = flattened_images.shape[1]
    centred_images = flattened_images - np.mean(flattened_images, axis=0) # subtract average from each
    covariance = centred_images.T @ centred_images
    covariance = covariance / (n-1)
    return covariance


def get_eigenvalues_and_eigenvectors(M: np.array, k: int, get_all=False, filename=None):
    if get_all:
        number_of_eigenvalues = M.shape[0]
    else:
        number_of_eigenvalues = k

    print("\tSaving matrix...")
    filename = save_matrix_for_deflation(M, filename=filename)
    print("\tSaved matrix to file {}".format(filename))

    print("\tDeflating...")
    e, v = d.deflate(filename, np.ones(M.shape[0]), number_of_eigenvalues)
    return np.array(e), np.array(v).T # return vectors as columns


def reduce_dimensions(images: np.array, eigenbase: np.array, k: int):
    return images @ (eigenbase[:, :k])


print("Reading images...")
images = read_images(Path(faces_path))

print("Calculating covariance...")
flattened_images = flatten_images(images)
covariance = get_covariance_matrix(flattened_images)

print("Calculating eigenvector base...")
e, V = get_eigenvalues_and_eigenvectors(covariance, 100, filename="amogus")
print(e, V)

print("Reducing dimensions...")
R = reduce_dimensions(flattened_images, V, 100)

plt.imshow(R[0].reshape(10,10))
plt.show()
