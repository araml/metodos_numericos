import numpy as np
import pandas as pd
import os
import seaborn as sns
from data_paths import csvs_path, figures_path
from matplotlib import pyplot as plt
from iterative_methods import *
from utils import *


def measure_iterations(iterative_method,
                       dimension: int,
                       repetitions: int,
                       low: int,
                       high: int,
                       diagonal_expansion_factor: int,
                       *args) -> list:
    all_iterations = []
    while len(all_iterations) < repetitions:
        x_0 = np.random.randint(low, high, dimension)
        try:
            m, _, b = create_test_case(dimension, low, high,
                                       diagonal_expansion_factor)
            _, iterations = iterative_method(m, b, x_0, *args)
            all_iterations.append(iterations)
        except:
            continue
    return all_iterations


def measure_iterations_growing_diagonal(iterative_method,
                                        dimension: int,
                                        repetitions: int,
                                        low: int,
                                        high: int,
                                        diagonal_expansion_factors: list,
                                        filename: str,
                                        *args) -> None:
    full_path = os.path.join(csvs_path, filename)
    if os.path.exists(full_path):
        os.remove(full_path)
    dict = {"factor": [], "iterations": []}
    for d in diagonal_expansion_factors:
        iterations = measure_iterations(
            iterative_method, dimension, repetitions, low, high, d, *args)
        for it in iterations:
            dict["factor"].append(d)
            dict["iterations"].append(it)
    df = pd.DataFrame(data=dict)
    df.to_csv(full_path, sep='\t', index=False)


def violin_plot_iterations(methods: list,
                           factors_to_plot: list,
                           figure_filename: str,
                           y_scale: str) -> None:
    method_data = read_method_datasets_from_csv(methods, factors_to_plot)
    plt.figure(figsize=(8,6))
    g = sns.violinplot(data=pd.concat(method_data), x="factor",
                       y="iterations", hue="method")
    g.set_yscale(y_scale)
    g.set_xlabel("Factor de expansión de la diagonal")
    g.set_ylabel("Cantidad de iteraciones hasta converger")
    plt.savefig(os.path.join(figures_path, figure_filename))
    plt.close()


def box_plot_iterations(methods: list,
                        factors_to_plot: list,
                        figure_filename: str,
                        y_scale: str = "linear"):
    method_data = read_method_datasets_from_csv(methods, factors_to_plot)
    plt.figure(figsize=(8,6))
    g = sns.boxplot(data=pd.concat(method_data), x="factor",
                    y="iterations", hue="method",
                    flierprops={"marker": '.'})
    g.set_yscale(y_scale)
    g.set_xlabel("Factor de expansión de la diagonal")
    g.set_ylabel("Cantidad de iteraciones hasta converger")
    plt.savefig(os.path.join(figures_path, figure_filename))
    plt.close()


def line_plot_iterations(methods: list,
                         factors_to_plot: list,
                         figure_filename: str,
                         scale: str = 'linear',
                         rotate_xticklabels: bool = False,
                         remove_outliers: bool = False,
                         low_q: float = 0.25,
                         high_q: float = 0.75) -> None:
    method_data = read_method_datasets_from_csv(
        methods, factors_to_plot, remove_outliers, low_q, high_q)
    means = []
    for df in method_data:
        method = df["method"].iloc[0]
        mean = df.groupby("factor").mean(numeric_only=True)
        means.append(mean.assign(method=method))

    g = sns.lineplot(pd.concat(means), x="factor", y="iterations",
                     hue="method", marker='o')
    g.set_xticks(means[0].axes[0].to_list())
    if rotate_xticklabels:
        g.tick_params(axis='x', rotation=90)
    plt.yscale(scale)
    plt.xlabel("Factor de expansión de la diagonal")
    plt.ylabel("Cantidad promedio de iteraciones hasta converger")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, figure_filename))
    plt.close()


def read_method_datasets_from_csv(methods: list,
                                  factors_to_plot: list,
                                  remove_outliers: bool = False,
                                  low_q: float = 0.25,
                                  high_q: float = 0.75) -> pd.DataFrame:
    method_data = []
    for method in methods:
        df = pd.read_csv(os.path.join(csvs_path, f"{method}_iterations.csv"),
                         sep='\t')
        df = df[df["factor"].isin(factors_to_plot)]
        # TODO: use a better method for detecting outliers
        if remove_outliers:
            df = df[df.groupby("factor").iterations.\
                transform(lambda x : (x < x.quantile(high_q)) & \
                          (x > x.quantile(low_q))).eq(1)]
        df["method"] = method
        method_data.append(df)
    return method_data


REPETITIONS = 100
DIMENSION = 100
LOW = 1
HIGH = 5
GS_SUM_RANGE = range(10, 1001)
GS_MATRIX_RANGE = range(10, 1001)
JACOBI_SUM_RANGE = range(120, 241)
JACOBI_MATRIX_RANGE = range(120, 241)
JACOBI_VS_GS_RANGE = range(120, 241, 5)
JACOBI_BOXPLOT_RANGE = range(120, 241, 15)

# === GAUSS-SEIDEL === #
measure_iterations_growing_diagonal(
    gauss_seidel_sum_method, DIMENSION, REPETITIONS, LOW, HIGH,
    GS_SUM_RANGE, "gs_sum_iterations.csv")

measure_iterations_growing_diagonal(
    gauss_seidel_matrix, DIMENSION, REPETITIONS, LOW, HIGH,
    GS_MATRIX_RANGE, "gs_matrix_iterations.csv")

violin_plot_iterations(
    ["gs_sum", "gs_matrix"], range(10, 30, 2),
    "violin_gs_iterations.png", "log")

box_plot_iterations(["gs_sum", "gs_matrix"], range(10, 30, 2),
                    "box_gs_iterations.png", "log")

line_plot_iterations(["gs_matrix", "gs_sum"], range(10, 31),
                     "line_gs_iterations.png")

line_plot_iterations(["gs_matrix", "gs_sum"], range(10, 31),
                     "line_gs_iterations_no_outliers.png", remove_outliers=True)

# === JACOBI === #
measure_iterations_growing_diagonal(
    jacobi_sum_method, DIMENSION, REPETITIONS, LOW, HIGH,
    JACOBI_SUM_RANGE, "jacobi_sum_iterations.csv")

measure_iterations_growing_diagonal(
    jacobi_matrix, DIMENSION, REPETITIONS, LOW, HIGH,
    JACOBI_MATRIX_RANGE, "jacobi_matrix_iterations.csv")

violin_plot_iterations(
    ["jacobi_sum", "jacobi_matrix"], JACOBI_BOXPLOT_RANGE,
    "violin_jacobi_iterations.png", "log")

box_plot_iterations(["jacobi_sum", "jacobi_matrix"], JACOBI_BOXPLOT_RANGE,
                    "box_jacobi_iterations.png", "log")

line_plot_iterations(["jacobi_matrix", "jacobi_sum"], range(120, 146),
                     "line_jacobi_iterations.png", rotate_xticklabels=True)

line_plot_iterations(["jacobi_matrix", "jacobi_sum"], range(120, 146),
                     "line_jacobi_iterations_no_outliers.png",
                     rotate_xticklabels=True, remove_outliers=True)

# === JACOBI/GS COMPARISON === #

violin_plot_iterations(
    ["jacobi_sum", "gs_sum"], JACOBI_BOXPLOT_RANGE,
    "violin_jacobi_vs_gs_sum_iterations.png", "log")

box_plot_iterations(
    ["jacobi_sum", "gs_sum"], JACOBI_BOXPLOT_RANGE,
    "box_jacobi_vs_gs_sum_iterations.png", "log")

line_plot_iterations(["jacobi_sum", "gs_sum"], JACOBI_VS_GS_RANGE,
                     "line_jacobi_vs_gs_iterations_sum.png",
                     rotate_xticklabels=True, remove_outliers=False,
                     scale="log")

violin_plot_iterations(
    ["jacobi_matrix", "gs_matrix"], JACOBI_BOXPLOT_RANGE,
    "violin_jacobi_vs_gs_matrix_iterations.png", "log")

box_plot_iterations(
    ["jacobi_matrix", "gs_matrix"], JACOBI_BOXPLOT_RANGE,
    "box_jacobi_vs_gs_matrix_iterations.png", "log")

line_plot_iterations(["jacobi_matrix", "gs_matrix"], JACOBI_VS_GS_RANGE,
                     "line_jacobi_vs_gs_iterations_matrix.png",
                     rotate_xticklabels=True, remove_outliers=False,
                     scale="log")
