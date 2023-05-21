import argparse
import matplotlib.pyplot as plt
import numpy as np
from figure_generation import *
from data_paths import *
from pathlib import Path
from utilities import * 
from PCA import PCA



# 2DPCA

def get_image_covariance_matrix(images: np.array) -> np.array:
    mean_pixel_values = np.mean(images, axis=0)
    centred_images = images - mean_pixel_values
    return np.mean(np.array([image.T @ image for image in centred_images]), axis=0)


def get_eigenbase_for_images(images: np.array,
                             k: int,
                             iters=10,
                             tolerance=1e-17,
                             filename=None) -> (np.array, np.array):
    G = get_image_covariance_matrix(images)
    return get_eigenvalues_and_eigenvectors(G, k, iters, tolerance, filename)


def get_feature_vectors(image: np.array, eigenbase: np.array, k: int) -> np.array:
    return image @ eigenbase[:, :k]


def reconstruct_image_2DPCA(feature_vectors: np.array, eigenbase: np.array, k: int) -> np.array:
    return feature_vectors @ eigenbase[:, :k].T


def compress_single_image_2DPCA(image: np.array, eigenbase: np.array, k: int) -> np.array:
    feature_vectors = get_feature_vectors(image, eigenbase, k)
    return reconstruct_image_2DPCA(feature_vectors, eigenbase, k)



parser = argparse.ArgumentParser("process_images")
parser.add_argument("--use_smaller_images",
                    help="Decrease image resolution for faster computation time",
                    action="store_true")
parser.add_argument("--scale_down_factor",
                    help="Factor by which to scale down image resolution",
                    type=int, default=2)
parser.add_argument("--number_of_eigenvectors",
                    help="Number of eigenvectors to compute",
                    type=int, default=100)
parser.add_argument("--number_of_components",
                    help="Number of components to use",
                    type=int, default=100)
parser.add_argument("--iterations",
                    help="Iterations for power method",
                    type=int, default=10)
parser.add_argument("--tolerance",
                    help="Tolerance for power method convergence criterion",
                    type=float, default=1e-17)

args = parser.parse_args()
number_of_eigenvectors = args.number_of_eigenvectors
number_of_components = args.number_of_components
iterations = args.iterations
tolerance = args.tolerance
scale_down_factor = args.scale_down_factor
print("number of eigenvectors: {}, iterations: {}, tolerance: {}".format(number_of_eigenvectors, iterations, tolerance))

print("Reading images...")
images = read_images(Path(faces_path), args.use_smaller_images, scale_down_factor)

pca = PCA(number_of_eigenvectors, iterations, tolerance)

covariance_matrix = pca.create_covariance_matrix(pca.flatten_images(images))
eigenvalues, _ = get_eigenvalues_and_eigenvectors(covariance_matrix, number_of_eigenvectors, iterations, tolerance)
plot_eigenvalues(eigenvalues, "{}_eigenvalues".format(eigenvalues.size))