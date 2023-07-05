import deflate
import numpy as np
import os
import pandas as pd
import parser
import seaborn as sns
from data_paths import *
from matplotlib import pyplot as plt
from pathlib import Path
from PCA import PCA
from PCA2D import PCA2D
from utilities import get_eigenvalues_and_eigenvectors, read_images

# Calculate error for ALL eigenvectors
def measure_eigenvector_errors(pca_class,
                               max_eigenvectors: int,
                               num_eigenvectors: int,
                               images: np.array,
                               iterations: int,
                               tolerance: float = 1e-17) -> None:
    pca_engine = pca_class(max_eigenvectors, iterations, tolerance)
    name = pca_engine.name

    # there's no way to do this in the PCA classes
    # without calculating all eigenvectors
    covariance_matrix = pca_engine.create_covariance_matrix(images)
    eigenvalues, eigenvectors = get_eigenvalues_and_eigenvectors(
        covariance_matrix, num_eigenvectors, iterations, tolerance)
    
    eigenvectors = eigenvectors.T
    matrix_eigenvectors = [covariance_matrix@eigenvectors[i]
                           for i in range(num_eigenvectors)]
    lambda_eigenvectors = [eigenvalues[i] * eigenvectors[i]
                           for i in range(num_eigenvectors)]
    diffs = [matrix_eigenvectors[i] - lambda_eigenvectors[i]
             for i in range(num_eigenvectors)]
    errors = np.array([np.linalg.norm(d) for d in diffs], dtype=np.float128)

    dict = {"error": errors, "eigenvalue": eigenvalues}
    dict["pca_class"] = name
    dict["eigenvector_number"] = np.arange(1, num_eigenvectors+1)
    dict["iterations"] = iterations
    dict["tolerance"] = tolerance

    file_path = Path(
        csvs_path,
        f"error_{name}_{iterations}its_tol{tolerance}.csv")
    if os.path.exists(file_path):
        os.remove(file_path)
    pd.DataFrame(dict).to_csv(file_path, sep='\t', index=False)
    

def read_csvs(pca_class_names,
              iteration_values: list,
              tolerance: float = 1e-17) -> pd.DataFrame:
    dfs = []
    for name in pca_class_names:
        for its in iteration_values:
            file_path = Path(
                csvs_path,
                f"error_{name}_{its}its_tol{tolerance}.csv")
            d = pd.read_csv(file_path, sep='\t')
            dfs.append(d)
    return pd.concat(dfs)


def plot_errors_for_eigenvector(pca_class_names: list,
                                iteration_values: list,
                                eigenvector_to_plot: int,
                                tolerance: float = 1e-17,
                                yscale: str = "log",
                                normalise_errors: bool = False) -> None:
    df = read_csvs(pca_class_names, iteration_values, tolerance)
    df = df[df["eigenvector_number"] == eigenvector_to_plot]
    if normalise_errors:
        df["error"] = df["error"] / np.abs(df["eigenvalue"])
    g = sns.lineplot(data=df, x="iterations", y="error",
                     hue="pca_class", marker='o')
    g.set(xticks=iteration_values)
    plt.yscale(yscale)
    plt.xlabel("Iteraciones del método de deflación")
    plt.ylabel(r"Error $\Vert Av - \lambda v \Vert_2$")
    plt.tight_layout()

    file_path = Path(figures_path,
        f"errors_eigenvector_{eigenvector_to_plot}_{'_'.join(pca_class_names)}" +
        f"_tol{tolerance}_{yscale}_normalise{normalise_errors}.png")
    plt.savefig(file_path)
    plt.close()


def plot_errors_for_many_eigenvectors(pca_class_names: list,
                                      iterations: int,
                                      eigenvectors_to_plot: list,
                                      tolerance: float = 1e-17,
                                      yscale: str ="log",
                                      normalise_errors: bool = False) -> None:
    df = read_csvs(pca_class_names, [iterations], tolerance)
    df = df[df["eigenvector_number"].isin(eigenvectors_to_plot)]
    if normalise_errors:
        df["error"] = df["error"] / np.abs(df["eigenvalue"])
    g = sns.lineplot(data=df, x="eigenvector_number", y="error",
                     hue="pca_class", marker='o')
    g.set(xticks=eigenvectors_to_plot)
    plt.yscale(yscale)
    plt.xlabel("Número de autovector calculado")
    plt.ylabel(r"Error $\Vert Av - \lambda v \Vert_2$")
    plt.tight_layout()

    file_path = Path(figures_path,
        f"errors_{iterations}its_{'_'.join(pca_class_names)}_tol{tolerance}" +
        f"_{yscale}_normalise{normalise_errors}.png")
    plt.savefig(file_path)
    plt.close()
    

p = parser.create_parser("error")
args = p.parse_args()

print("Reading images...")
images = read_images(Path(faces_path), 1)
max_eigenvectors_PCA = images[0].size
max_eigenvectors_2DPCA = images[0].shape[1]

iterations = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]

for its in iterations:
    measure_eigenvector_errors(
        PCA, max_eigenvectors_PCA, 1200, images, its)
    measure_eigenvector_errors(
        PCA2D, max_eigenvectors_2DPCA, max_eigenvectors_2DPCA, images, its)

    plot_errors_for_many_eigenvectors(
        ["1DPCA", "2DPCA"], its, range(10, 91, 10), normalise_errors=True)
    plot_errors_for_many_eigenvectors(
        ["1DPCA"], its, range(100, 1200, 100), normalise_errors=True)
    plot_errors_for_many_eigenvectors(
        ["1DPCA", "2DPCA"], its, range(10, 91, 10), normalise_errors=False)
    plot_errors_for_many_eigenvectors(
        ["1DPCA"], its, range(100, 1200, 100), normalise_errors=False)
    
for v in range(1, 11):
    plot_errors_for_eigenvector(
        ["1DPCA", "2DPCA"], iterations, v, normalise_errors=True)
    plot_errors_for_eigenvector(
        ["1DPCA", "2DPCA"], iterations, v, normalise_errors=False)

for v in [410, 600, 1000]:
    plot_errors_for_eigenvector(
        ["1DPCA"], iterations, v, normalise_errors=True)
    plot_errors_for_eigenvector(
        ["1DPCA"], iterations, v, normalise_errors=False)