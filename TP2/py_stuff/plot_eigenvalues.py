import numpy as np
import pandas as pd
import parser
from seaborn import lineplot, scatterplot
from data_paths import csvs_path, faces_path
from figure_generation import plot_eigenvalues
from pathlib import Path
from PCA import PCA, PCABase
from PCA2D import PCA2D
from utilities import read_images

def create_eigenvalue_csv(pca_class,
                          images: np.array,
                          number_of_eigenvectors: int,
                          iterations: int,
                          tolerance: float) -> None:
    pca_engine = pca_class(number_of_eigenvectors, iterations, tolerance)
    pca_engine.fit(images)
    name = pca_engine.name
    dict = {"eigenvalue": pca_engine.eigenvalues,
            "eigenvalue_number": range(1, number_of_eigenvectors+1)}
    df = pd.DataFrame(dict)
    df["iterations"] = iterations
    df["tolerance"] = tolerance
    df["pca_class"] = name
    file_path = Path(
        csvs_path, f"eigenvalues_{iterations}its_tol{tolerance}_{name}.csv")
    df.to_csv(file_path, sep='\t', index=False)

def run_plot_eigenvalues_experiment() -> None:
    p = parser.create_parser("plot_eigenvalues")
    args = p.parse_args()
    iterations = args.iterations
    tolerance = args.tolerance

    print("Reading images...")
    images = read_images(Path(faces_path), args.scale_down_factor)

    # create_eigenvalue_csv(PCA, images, 1000, iterations, tolerance)
    # create_eigenvalue_csv(PCA2D, images, 92, iterations, tolerance)

    plot_eigenvalues(["1DPCA"], range(1, 1001), iterations,
                     plot_function=scatterplot, yscale="symlog",
                     kwargs = {"s": 2, "linewidth": 0})
    plot_eigenvalues(["1DPCA", "2DPCA"], range(1, 92, 5), iterations,
                     plot_function=lineplot, yscale="linear", set_xticks=True)
    plot_eigenvalues(["1DPCA", "2DPCA"], range(1, 11), iterations,
                     plot_function=lineplot, yscale="linear", set_xticks=True)

if __name__ == '__main__': 
    run_plot_eigenvalues_experiment()
