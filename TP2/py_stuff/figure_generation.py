from data_paths import *
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from PCA import PCA, PCABase
from random import choices
from string import ascii_letters


def plot_eigenvalues(eigenvalues: np.array, filename: str):
    file_path = Path(figures_path, filename + '.png')
    x = np.arange(1, eigenvalues.size+1)
    plt.plot(x, eigenvalues, '-o')
    plt.xticks(x)
    plt.xlabel("Número de componente")
    plt.ylabel("Autovalor")
    plt.title("{} primeros autovalores".format(eigenvalues.size))
    plt.savefig(file_path)
    return file_path


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


def create_pca_image_comparison(pca_engine: PCABase,
                                images: np.array,
                                image_index: int,
                                figsize: (int, int),
                                colourmap=plt.cm.viridis) -> str:
    h, w = images.shape[1], images.shape[2]
    original_image, compressed_image = pca_engine.get_image_comparison(images, image_index)
    filename = "image_comparison_{}_{}components_{}_{}".format(image_index, pca_engine.k, colourmap.name, pca_engine.name)
    return save_image_comparison(original_image, compressed_image, h, w, figsize, filename, colourmap)


def create_corrcoef_figure(pca_engine: PCABase,
                           images: np.array,
                           colourmap=plt.cm.GnBu):
    centred_images = pca_engine.centre_images(images)
    flattened_compressed_images = pca_engine.flatten_images(pca_engine.transform(images))
    similarity = np.corrcoef(centred_images, flattened_compressed_images)
    
    filename = "corrcoef_{}components_{}_{}".format(pca_engine.k, colourmap.name, pca_engine.name)
    file_path = Path(figures_path, filename + '.png')

    plt.pcolor(similarity, cmap=colourmap)
    cb = plt.colorbar()
    plt.savefig(file_path)
    cb.remove()