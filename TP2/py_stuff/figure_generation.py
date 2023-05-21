from data_paths import *
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from PCA import PCA
from random import choices
from string import ascii_letters

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