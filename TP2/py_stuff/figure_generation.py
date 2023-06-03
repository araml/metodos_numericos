from data_paths import *
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from PCA import PCA, PCABase
from PCA2D import PCA2D
from utilities import centre_images


def plot_eigenvalues(images: np.array,
                     number_of_eigenvectors: int,
                     iterations: int = 10,
                     tolerance: float = 1e-17):
    pca = PCA(number_of_eigenvectors, iterations, tolerance)
    pca.fit(images)
    eigenvalues = pca.eigenvalues
    pca_2d = PCA2D(number_of_eigenvectors, iterations, tolerance)
    pca_2d.fit(images)
    eigenvalues_2d = pca_2d.eigenvalues
    x = np.arange(1, eigenvalues.size+1)
    plt.plot(x, eigenvalues, label="PCA")
    plt.plot(x, eigenvalues_2d, label="2DPCA")
    plt.xticks(np.arange(1, eigenvalues.size+1, 10))
    plt.xlabel("Número de componente")
    plt.ylabel("Autovalor")
    plt.yscale("log")
    plt.title(f"{eigenvalues.size} primeros autovalores, {iterations} iteraciones, escala logarítmica")
    plt.legend()
    file_path = Path(figures_path, f"{eigenvalues.size}autovalores_{iterations}its_log.png")
    plt.savefig(file_path)
    plt.clf()


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


def create_compression_grid(pca_class,
                            images: np.array,
                            image_indexes: list,
                            ks: int,
                            iterations: int,
                            tolerance: float,
                            subplots_height: int,
                            subplots_width: int,
                            figsize: (int, int),
                            colourmap = plt.cm.viridis) -> None:
    assert(len(ks) == subplots_height * subplots_width)
    h, w = images[0].shape
    max_k = max(ks)
    pca = pca_class(max_k, iterations, tolerance)
    pca.fit(images)
    for image_index in image_indexes:
        _, axs = plt.subplots(subplots_height, subplots_width, figsize=figsize)
        for i, ax in enumerate(axs.flatten()):
            pca.set_components_dimension(ks[i])
            compressed_image = pca.transform(images)[image_index]
            ax.imshow(compressed_image.reshape(h, w), cmap=colourmap)
            ax.set_title(f"{ks[i]} componentes")
            ax.axis('off')
        plt.tight_layout()
        filename = f"compression_{image_index}_{subplots_height}x{subplots_width}_{iterations}iterations_{pca.name}_{colourmap.name}.png"
        plt.savefig(Path(figures_path, filename))
        plt.clf()


def create_corrcoef_baseline(images: np.array, colourmap=plt.cm.GnBu):
    similarity = np.corrcoef(centre_images(images))
    
    filename = f"corrcoef_baseline_{colourmap.name}.png"
    file_path = Path(figures_path, filename)

    plt.pcolor(similarity, cmap=colourmap)
    plt.colorbar()
    plt.savefig(file_path)
    plt.clf()


def create_compression_corrcoef_figures(pca_class,
                                        images: np.array,
                                        ks: list,
                                        iterations: int = 10,
                                        tolerance: float = 1e-17,
                                        colourmap = plt.cm.GnBu) -> None:
    max_k = max(ks)
    pca = pca_class(max_k, iterations, tolerance)
    pca.fit(images)
    for k in ks:
        pca.set_components_dimension(k)
        centred_projected_images = centre_images(pca.project_images(images))
        similarity = np.corrcoef(centred_projected_images)
        filename = f"corrcoef_{k}components_{colourmap.name}_{pca.name}.png"
        file_path = Path(figures_path, filename)

        plt.pcolor(similarity, cmap=colourmap)
        plt.colorbar()
        plt.savefig(file_path)
        plt.close()