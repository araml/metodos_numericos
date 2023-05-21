import argparse
import matplotlib.pyplot as plt
import numpy as np
from image_paths import * # TODO: rename image_paths since it also contains matrix path now
from pathlib import Path
from random import choices
from string import ascii_letters
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


# Figures

def save_eigenvector_figure(eigenvectors: np.array,
                            image_height: int,
                            image_width: int,
                            subplots_height: int,
                            subplots_width: int,
                            figsize: (int, int),
                            filename=None,
                            colourmap=plt.cm.viridis) -> str:
    if not filename:
        filename = ''.join(choices(ascii_letters, k=12))
    file_path = Path(figures_path, filename + '.png')

    _, axs = plt.subplots(subplots_height, subplots_width, figsize=figsize)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(eigenvectors[:,i].reshape(image_height, image_width), cmap=colourmap)
        ax.set_title("Autovector {}".format(i+1))
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(file_path)
    return file_path


def save_image_comparison(original_image: np.array,
                          compressed_image: np.array,
                          image_height: int,
                          image_width: int,
                          figsize: (int, int),
                          filename=None,
                          colourmap=plt.cm.viridis) -> str:
    if not filename:
        filename = ''.join(choices(ascii_letters, k=12))
    file_path = Path(figures_path, filename + '.png')
    _, axs = plt.subplots(1, 2, figsize=figsize)

    axs[0].imshow(original_image.reshape(image_height, image_width), cmap=colourmap)
    axs[0].set_title("Imagen original")
    axs[0].axis('off')

    axs[1].imshow(compressed_image.reshape(image_height, image_width), cmap=colourmap)
    axs[1].set_title("Imagen comprimida")
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig(file_path)
    return file_path


def create_pca_image_comparison(pca: PCA,
                                images: np.array,
                                image_index: int,
                                number_of_components: int,
                                figsize: (int, int),
                                colourmap=plt.cm.viridis) -> str:
    h, w = images.shape[1], images.shape[2]
    original_image, compressed_image = pca.get_image_comparison(images, image_index, number_of_components)
    filename = "image_comparison_{}_{}components_{}".format(image_index, number_of_components, colourmap.name)
    return save_image_comparison(original_image, compressed_image, h, w, figsize, filename, colourmap)


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
h, w = images.shape[1], images.shape[2]

pca = PCA(number_of_eigenvectors, iterations, tolerance)
pca.fit(images, "amogus")

create_pca_image_comparison(pca, images, 120, number_of_components, (12,12), plt.cm.magma)
create_pca_image_comparison(pca, images, 130, number_of_components, (12,12), plt.cm.magma)
create_pca_image_comparison(pca, images, 140, number_of_components, (12,12), plt.cm.magma)
create_pca_image_comparison(pca, images, 150, number_of_components, (12,12), plt.cm.magma)
create_pca_image_comparison(pca, images, 160, number_of_components, (12,12), plt.cm.magma)
create_pca_image_comparison(pca, images, 170, number_of_components, (12,12), plt.cm.magma)
create_pca_image_comparison(pca, images, 180, number_of_components, (12,12), plt.cm.magma)
