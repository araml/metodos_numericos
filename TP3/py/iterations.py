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


def violin_plot_iterations(factors_to_plot: list,
                           csv_filename: str,
                           figure_filename: str,
                           y_scale: str) -> None:
    df = pd.read_csv(os.path.join(csvs_path, csv_filename), sep='\t')
    df = df[df["factor"].isin(factors_to_plot)]
    plt.figure(figsize=(8,6))
    g = sns.violinplot(df, x="factor", y="iterations", color="tab:orange")
    g.set_yscale(y_scale)
    g.set_xlabel("Factor de expansión de la diagonal")
    g.set_ylabel("Cantidad de iteraciones hasta converger")
    plt.savefig(os.path.join(figures_path, figure_filename))
    plt.close()


def box_plot_iterations(factors_to_plot: list,
                        csv_filename: str,
                        figure_filename: str,
                        y_scale: str = "linear"):
    df = pd.read_csv(os.path.join(csvs_path, csv_filename), sep='\t')
    df = df[df["factor"].isin(factors_to_plot)]
    plt.figure(figsize=(8,6))
    g = sns.boxplot(df, x="factor", y="iterations", color="tab:orange",
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
    datasets = []
    for method in methods:
        df = pd.read_csv(os.path.join(csvs_path, f"{method}_iterations.csv"),
                         sep='\t')
        df = df[df["factor"].isin(factors_to_plot)]
        if remove_outliers:
            df = df[df.groupby("factor").iterations.\
                transform(lambda x : (x < x.quantile(high_q)) & \
                          (x > x.quantile(low_q))).eq(1)]
        iteration_means = df.groupby("factor").mean()
        datasets.append(iteration_means.assign(method=method))

    g = sns.lineplot(pd.concat(datasets), x="factor", y="iterations",
                     hue="method", marker='o')
    g.set_xticks(datasets[0].axes[0].to_list())
    if rotate_xticklabels:
        g.tick_params(axis='x', rotation=90)
    plt.yscale(scale)
    plt.xlabel("Factor de expansión de la diagonal")
    plt.ylabel("Cantidad promedio de iteraciones hasta converger")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, figure_filename))
    plt.close()


REPETITIONS = 100
DIMENSION = 100
LOW = 1
HIGH = 5
GS_SUM_RANGE = range(10, 1001)
GS_MATRIX_RANGE = range(10, 1001)
JACOBI_SUM_RANGE = range(120, 241)
JACOBI_MATRIX_RANGE = range(120, 241)

GS_BOX_RANGES = [range(15, 21), range(20, 26), range(50, 56), range(100, 106)]

# === GAUSS-SEIDEL SUM === #
measure_iterations_growing_diagonal(
    gauss_seidel_sum_method, DIMENSION, REPETITIONS, LOW, HIGH,
    GS_SUM_RANGE, "gs_sum_iterations.csv")

violin_plot_iterations(
    range(10, 16), "gs_sum_iterations.csv",
    "violin_gs_sum_iterations_10_15.png", "log")

box_plot_iterations(range(10, 16), "gs_sum_iterations.csv",
                    "box_gs_sum_iterations_10_15.png", "log")

for rg in GS_BOX_RANGES:
    violin_plot_iterations(rg, "gs_sum_iterations.csv",
                           f"violin_gs_sum_iterations_{rg[0]}_{rg[-1]}.png",
                           "linear")
    box_plot_iterations(rg, "gs_sum_iterations.csv",
                        f"box_gs_sum_iterations_{rg[0]}_{rg[-1]}.png", "linear")


# === GAUSS-SEIDEL MATRIX === #
measure_iterations_growing_diagonal(
    gauss_seidel_matrix, DIMENSION, REPETITIONS, LOW, HIGH,
    GS_MATRIX_RANGE, "gs_matrix_iterations.csv")

violin_plot_iterations(
    range(10, 16), f"gs_matrix_iterations.csv",
    "gs_matrix_iterations_10_15.png", "log")

box_plot_iterations(range(10, 16), "gs_matrix_iterations.csv",
                    "box_gs_matrix_iterations_10_15.png", "log")

for rg in GS_BOX_RANGES:
    violin_plot_iterations(rg, "gs_matrix_iterations.csv",
                           f"violin_gs_matrix_iterations_{rg[0]}_{rg[-1]}.png",
                           "linear")
    box_plot_iterations(rg, "gs_matrix_iterations.csv",
                        f"box_gs_matrix_iterations_{rg[0]}_{rg[-1]}.png", "linear")

line_plot_iterations(["gs_matrix", "gs_sum"], range(10, 26),
                     "line_gs_iterations.png")
    
line_plot_iterations(["gs_matrix", "gs_sum"], range(10, 26),
                     "line_gs_iterations_no_outliers.png", remove_outliers=True)

# === JACOBI SUM === #
measure_iterations_growing_diagonal(
    jacobi_sum_method, DIMENSION, REPETITIONS, LOW, HIGH,
    JACOBI_SUM_RANGE, "jacobi_sum_iterations.csv")

violin_plot_iterations(
    range(120, 126), "jacobi_sum_iterations.csv",
    "violin_jacobi_sum_iterations_120_125.png", "linear")

violin_plot_iterations(
    range(120, 145, 5), "jacobi_sum_iterations.csv",
    "violin_jacobi_sum_iterations_120_140.png", "linear")

box_plot_iterations(
    range(120, 126), "jacobi_sum_iterations.csv",
    "box_jacobi_sum_iterations_120_125.png", "linear")

box_plot_iterations(
    range(120, 145, 5), "jacobi_sum_iterations.csv",
    "box_jacobi_sum_iterations_120_140.png", "linear")

# == JACOBI MATRIX == #
measure_iterations_growing_diagonal(
    jacobi_matrix, DIMENSION, REPETITIONS, LOW, HIGH,
    JACOBI_MATRIX_RANGE, "jacobi_matrix_iterations.csv")

violin_plot_iterations(
    range(120, 126), "jacobi_matrix_iterations.csv",
    "violin_jacobi_matrix_iterations_120_125.png", "linear")

violin_plot_iterations(
    range(120, 145, 5), "jacobi_matrix_iterations.csv",
    "violin_jacobi_matrix_iterations_120_140.png", "linear")

box_plot_iterations(
    range(120, 126), "jacobi_matrix_iterations.csv",
    "box_jacobi_matrix_iterations_120_125.png", "linear")

box_plot_iterations(
    range(120, 145, 5), "jacobi_matrix_iterations.csv",
    "box_jacobi_matrix_iterations_120_140.png", "linear")

line_plot_iterations(["jacobi_matrix", "jacobi_sum"], range(120, 146),
                     "line_jacobi_iterations.png", rotate_xticklabels=True)

line_plot_iterations(["jacobi_matrix", "jacobi_sum"], range(120, 146),
                     "line_jacobi_iterations_no_outliers.png",
                     rotate_xticklabels=True, remove_outliers=True)