import numpy as np
import pandas as pd
import seaborn as sns
from data_paths import *
from matplotlib import pyplot as plt
from pathlib import Path
from PCA import PCA, PCABase
from PCA2D import PCA2D
from utilities import centre_images


def plot_eigenvalues(pca_class_names: list,
                     eigenvalue_numbers_to_plot: list,
                     iterations: int = 20,
                     tolerance: float = 1e-17,
                     yscale: str = "log",
                     plot_function = sns.lineplot,
                     set_xticks = False,
                     kwargs = {}):
    dfs = []
    for name in pca_class_names:
        file_path = Path(
            csvs_path, f"eigenvalues_{iterations}its_tol{tolerance}_{name}.csv")
        df = pd.read_csv(file_path, sep='\t')
        df = df[df["eigenvalue_number"].isin(eigenvalue_numbers_to_plot)]
        dfs.append(df)
    df = pd.concat(dfs)
    g = plot_function(data=df, x="eigenvalue_number", y="eigenvalue",
                      hue="pca_class", marker='o', **kwargs)
    if set_xticks:
        g.set(xticks=eigenvalue_numbers_to_plot)
    plt.xlabel("Número de componente")
    plt.ylabel("Autovalor")
    plt.yscale(yscale)
    plt.tight_layout()

    file_path = Path(
        figures_path,
        f"{max(eigenvalue_numbers_to_plot)}autovalores_{iterations}its_" +
        f"{'_'.join(pca_class_names)}.png")
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
        filename = f"compression_{image_index}_{subplots_height}x{subplots_width}" + \
        f"_{iterations}iterations_{pca.name}_maxk{max(ks)}_{colourmap.name}.png"
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