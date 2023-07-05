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
from utilities import measure_execution_time, read_images

def measure_time_for_num_eigenvectors(pca_class,
                                      images: np.array,
                                      repetitions: int,
                                      num_eigenvectors: int,
                                      iterations: int = 10,
                                      tolerance: float = 1e-17) -> pd.DataFrame:
    dict = {"execution_time": []}
    pca_engine = pca_class(num_eigenvectors, iterations, tolerance)
    for i in range(repetitions):
        if i % 10 == 0:
            print(f"REPETITION {i}")
        t = measure_execution_time(pca_engine.fit, images)
        dict["execution_time"].append(t)
    dict["pca_class"] = pca_engine.name
    dict["num_eigenvectors"] = num_eigenvectors
    dict["iterations"] = iterations
    dict["tolerance"] = tolerance
    return pd.DataFrame(dict)


def measure_time_growing_k(pca_class,
                           images: np.array,
                           repetitions: int,
                           values_to_measure: list,
                           iterations: int = 10,
                           tolerance: float = 1e-17):
    dfs = []
    for num_eigenvectors in values_to_measure:
        r = measure_time_for_num_eigenvectors(
            pca_class, images, repetitions, num_eigenvectors,
            iterations, tolerance)
        dfs.append(r)
    e = pca_class(num_eigenvectors, iterations, tolerance)
    filename = f"execution_times_{e.name}_{iterations}its.csv"
    file_path = Path(csvs_path, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    df = pd.concat(dfs)
    df.to_csv(file_path, sep='\t', index=False)


def plot_execution_time(pca_class_names: list,
                        iterations: int = 10,
                        remove_outliers: bool = True,
                        yscale = "linear"):
    dfs = []
    for name in pca_class_names:
        file_path = Path(csvs_path, f"execution_times_{name}_{iterations}its.csv")
        df = pd.read_csv(file_path, sep='\t')
        if remove_outliers:
            by_num_eigenvectors = df.groupby("num_eigenvectors")
            means = by_num_eigenvectors.execution_time.transform("mean")
            stds = by_num_eigenvectors.execution_time.transform("std")
            df = df[df.execution_time.between(means - stds*3, means + stds*3)]
        dfs.append(df)
    
    means = []
    for df in dfs:
        pca_class = df["pca_class"].iloc[0]
        mean = df.groupby("num_eigenvectors").mean(numeric_only=True)
        means.append(mean.assign(pca_class=pca_class))

    g = sns.lineplot(pd.concat(means), x="num_eigenvectors", y="execution_time", hue="pca_class", marker='o')
    plt.yscale(yscale)
    plt.xlabel("Autovectores calculados")
    plt.ylabel("Tiempo promedio de ejecución")
    plt.tight_layout()
    filename = f"execution_time_{'_'.join(pca_class_names)}_{yscale}_{iterations}its.png"
    plt.savefig(Path(figures_path, filename))
    plt.close()


p = parser.create_parser("execution_time")
args = p.parse_args()

print("Reading images...")
images = read_images(Path(faces_path), args.scale_down_factor)

REPETITIONS = 50

measure_time_growing_k(PCA, images, REPETITIONS, range(1, 93, 13))
measure_time_growing_k(PCA2D, images, REPETITIONS, range(1, 93, 13))

plot_execution_time(["1DPCA", "2DPCA"], yscale="linear")
plot_execution_time(["2DPCA"], yscale="log")