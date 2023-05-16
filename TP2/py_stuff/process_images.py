import argparse
import deflate as d
import matplotlib.pyplot as plt
import numpy as np
from image_paths import * # TODO: rename image_paths since it also contains matrix path now
from pathlib import Path
from random import choices
from string import ascii_letters


# Common functions

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


def read_images(path_to_images, use_smaller_images, scale_down_factor=1) -> list:
    paths = []
    images = []

    for path in sorted(list(path_to_images.rglob('*/*.pgm'))):
        paths.append(path)
        image = (plt.imread(path))
        if use_smaller_images:
            image = image[::scale_down_factor,::scale_down_factor] / 255
        images.append(image)

    return images


def get_eigenvalues_and_eigenvectors(M: np.array,
                                     k: int,
                                     iters=10,
                                     tolerance=1e-17,
                                     filename=None,
                                     get_all=False) -> (np.array, np.array):
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


# 1DPCA

def flatten_images(images: np.array) -> np.array:
    square_images = np.stack(images)
    flattened_images = square_images.reshape(square_images.shape[0], square_images[0].size)
    return flattened_images


def get_1d_covariance_matrix(flattened_images: np.array) -> np.array:
    n = flattened_images.shape[1]
    centred_images = flattened_images - np.mean(flattened_images, axis=0) # subtract average from each
    covariance = centred_images.T @ centred_images
    covariance = covariance / (n-1)
    return covariance


def reduce_dimensions(flattened_images: np.array, eigenbase: np.array, k: int) -> np.array:
    return flattened_images @ (eigenbase[:, :k])


# 2DPCA

def get_image_covariance_matrix(images: np.array) -> np.array:
    mean_pixel_values = np.mean(images, axis=0)
    centred_images = images - mean_pixel_values
    return np.mean(np.array([image.T @ image for image in centred_images]), axis=0)


def get_eigenbase_for_images(images: np.array,
                             k: int,
                             iters=10,
                             tolerance=1e-17,
                             filename=None,
                             get_all=False) -> (np.array, np.array):
    G = get_image_covariance_matrix(images)
    return get_eigenvalues_and_eigenvectors(G, k, iters, tolerance, filename, get_all)


def get_feature_vectors(image: np.array, eigenbase: np.array, k: int) -> np.array:
    return image @ eigenbase[:, :k]


parser = argparse.ArgumentParser("process_images")
parser.add_argument("use_smaller_images",
                    help="Decrease image resolution for faster computation time",
                    type=bool)
parser.add_argument("scale_down_factor",
                    help="Factor by which to scale down image resolution",
                    type=int)

args = parser.parse_args()

print("Reading images...")
images = read_images(Path(faces_path), args.use_smaller_images, args.scale_down_factor)

print("Calculating image feature vectors...")
eigenvalues, eigenbase = get_eigenbase_for_images(images, 0, get_all=True, filename="amogus")
feature_vectors = get_feature_vectors(images[0], eigenbase, 46)
plt.imshow(feature_vectors)
plt.show()

# print("Calculating covariance...")
# flattened_images = flatten_images(images)
# covariance = get_1d_covariance_matrix(flattened_images)

# print("Calculating eigenvector base...")
# e, V = get_eigenvalues_and_eigenvectors(covariance, 100, filename="amogus")
# print(e, V)

# print("Reducing dimensions...")
# R = reduce_dimensions(flattened_images, V, 100)

# plt.imshow(R[0].reshape(10,10))
# plt.show()
