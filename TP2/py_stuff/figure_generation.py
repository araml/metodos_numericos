from data_paths import *
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from PCA import PCA, PCABase
from random import choices
from string import ascii_letters
from utilities import centre_images, flatten_images


def plot_eigenvalues(pca_engine: PCABase):
    eigenvalues = pca_engine.eigenvalues
    x = np.arange(1, eigenvalues.size+1)
    plt.plot(x, eigenvalues)
    # plt.xticks(x)
    plt.xlabel("Número de componente")
    plt.ylabel("Autovalor")
    plt.title(f"{eigenvalues.size} primeros autovalores, {pca_engine.name}")
    file_path = Path(figures_path, f"{eigenvalues.size}autovalores_{pca_engine.name}.png")
    plt.savefig(file_path)
    plt.clf()
    return file_path


def save_eigenvector_figure(pca_engine: PCABase,
                            subplots_height: int,
                            subplots_width: int,
                            figsize: (int, int),
                            colourmap=plt.cm.viridis) -> str:
    eigenfaces = pca_engine.get_eigenfaces()
    _, axs = plt.subplots(subplots_height, subplots_width, figsize=figsize)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(eigenfaces[i], cmap=colourmap)
        ax.set_title("Autovector {}".format(i+1))
        ax.axis('off')
    plt.tight_layout()
    file_path = Path(figures_path,
        f"{subplots_height*subplots_width}eigenfaces_{pca_engine.name}_{colourmap.name}.png")
    plt.savefig(file_path)
    plt.clf()
    return file_path


def save_image_comparison(original_image: np.array,
                          compressed_image: np.array,
                          image_height: int,
                          image_width: int,
                          figsize: (int, int),
                          filename=None,
                          colourmap=plt.cm.viridis,
                          show_image=False) -> str:
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
    if show_image:
        plt.show()
    plt.clf()
    return file_path


def create_pca_image_comparison(pca_engine: PCABase,
                                images: np.array,
                                image_index: int,
                                figsize: (int, int) = (8, 8),
                                colourmap = plt.cm.viridis,
                                show_image: bool = False) -> str:
    h, w = images.shape[1], images.shape[2]
    original_image, compressed_image = pca_engine.get_image_comparison(images, image_index)
    filename = "image_comparison_{}_{}components_{}_{}".format(image_index,
    pca_engine.k, colourmap.name, pca_engine.name)
    return save_image_comparison(original_image, compressed_image, h, w, figsize, filename,
                                 colourmap, show_image)


def create_corrcoef_figure(pca_engine: PCABase,
                           images: np.array,
                           colourmap=plt.cm.GnBu):
    centred_images = centre_images(images)
    flattened_compressed_images = flatten_images(pca_engine.transform(images))
    similarity = np.corrcoef(centred_images, flattened_compressed_images)
    
    filename = "corrcoef_{}components_{}_{}".format(pca_engine.k, colourmap.name, pca_engine.name)
    file_path = Path(figures_path, filename + '.png')

    plt.pcolor(similarity, cmap=colourmap)
    plt.colorbar()
    plt.savefig(file_path)
    plt.clf()


def create_compression_corrcoef_figures(pca_class,
                                        images: np.array,
                                        small_k: int,
                                        large_k: int,
                                        iterations: int = 10,
                                        tolerance: float = 1e-17,
                                        colourmap = plt.cm.GnBu) -> None:
    assert(large_k > small_k and small_k > 0)

    pca_engine = pca_class(large_k, iterations, tolerance)
    pca_engine.fit(images)

    create_corrcoef_figure(pca_engine, images, colourmap)

    pca_engine.set_components_dimension(small_k)
    create_corrcoef_figure(pca_engine, images, colourmap)