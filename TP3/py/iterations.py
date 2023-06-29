import numpy as np
import pandas as pd
import os
import seaborn as sns
from data_paths import csvs_path, figures_path
from iterative_methods import *
from matplotlib import pyplot as plt
from utils import *


def measure_iterations(iterative_method_name,
                       dimension: int,
                       repetitions: int,
                       low: int,
                       high: int,
                       diagonal_expansion_factor: int,
                       iterations: int = 10000,
                       eps: float = 1e-6) -> pd.DataFrame:
    dict = {"iterations": [], "spectral_radius": [],
            "condition_number": []}
    iterative_method = methods_by_name[iterative_method_name]

    while len(dict["iterations"]) < repetitions:
        x_0 = np.random.randint(low, high, dimension)
        try:
            m, _, b = create_test_case(dimension, low, high,
                                       diagonal_expansion_factor)
            _, its = iterative_method(m, b, x_0, iterations, eps)
            dict["iterations"].append(its)
            t = iteration_matrix(m, iterative_method_name)
            dict["spectral_radius"].append(spectral_radius(t))
            dict["condition_number"].append(np.linalg.cond(m))
            r = len(dict["iterations"])
            if r % 10 == 0:
                print(f"\tFinished repetition {r}")
        except:
            continue

    dict["factor"] = diagonal_expansion_factor
    dict["method"] = iterative_method_name
    dict["epsilon"] = eps
    dict["dimension"] = dimension

    return pd.DataFrame(data=dict)


def measure_iterations_growing_diagonal(iterative_method_name,
                                        dimension: int,
                                        repetitions: int,
                                        low: int,
                                        high: int,
                                        diagonal_expansion_factors: list,
                                        iterations: int = 10000,
                                        eps: float = 1e-6) -> None:
    full_path = os.path.join(csvs_path,
        f"{iterative_method_name}_iterations_dim{dimension}.csv")
    if os.path.exists(full_path):
        os.remove(full_path)

    results = []
    for d in diagonal_expansion_factors:
        print(f"Calculating {iterative_method_name}, d = {d}")
        r = measure_iterations(
            iterative_method_name, dimension, repetitions,
            low, high, d, iterations, eps)
        results.append(r)

    df = pd.concat(results)
    df.to_csv(full_path, sep='\t', index=False)


def box_plot_iterations(methods: list,
                        factors_to_plot: list,
                        dimension: int,
                        figure_filename: str,
                        y_scale: str = "linear"):
    method_data = read_method_datasets_from_csv(methods, factors_to_plot,
                                                dimension)
    plt.figure(figsize=(8,6))
    g = sns.boxplot(data=pd.concat(method_data), x="factor",
                    y="iterations", hue="method",
                    flierprops={"marker": '.'})
    g.set_yscale(y_scale)
    g.set_xlabel("Factor de expansión de la diagonal")
    g.set_ylabel("Cantidad de iteraciones hasta converger")
    plt.savefig(os.path.join(figures_path,
                             f"{figure_filename}_dim{dimension}.png"))
    plt.close()


def line_plot_iterations(methods: list,
                         factors_to_plot: list,
                         dimension: int,
                         figure_filename: str,
                         scale: str = 'linear',
                         rotate_xticklabels: bool = False,
                         remove_outliers: bool = False) -> None:
    method_data = read_method_datasets_from_csv(
        methods, factors_to_plot, dimension, remove_outliers)
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
    plt.savefig(os.path.join(figures_path,
                             f"{figure_filename}_dim{dimension}.png"))
    plt.close()


def plot_spectral_radius_iterations(methods: list,
                                    factors_to_plot: list,
                                    dimension: int,
                                    figure_filename: str,
                                    scale: str = 'linear',
                                    sample_frac: float = 0.25) -> None:
    method_data = read_method_datasets_from_csv(
        methods, factors_to_plot, dimension)
    df = pd.concat(method_data)
    sampled = df.sample(frac=sample_frac)
    plt.figure(figsize=(6,6))
    g = sns.scatterplot(sampled, x="spectral_radius", y="iterations",
                        hue="method", alpha=0.3, edgecolor="none")
    plt.yscale(scale)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path,
                             f"{figure_filename}_dim{dimension}.png"))
    plt.close()


def read_method_datasets_from_csv(methods: list,
                                  factors_to_plot: list,
                                  dimension: int,
                                  remove_outliers: bool = False) -> list:
    method_data = []
    for method in methods:
        full_path = os.path.join(csvs_path,
            f"{method}_iterations_dim{dimension}.csv")
        df = pd.read_csv(full_path, sep='\t')
        df = df[df["factor"].isin(factors_to_plot)]
        if remove_outliers:
            # reference: https://stackoverflow.com/questions/57256870/flag-outliers-in-the-dataframe-for-each-group
            # Remove rows where the iterations are three standard deviations
            # above or below the mean for their respective factor.
            # This is far more robust than using quantiles, which was leaving
            # the dataframe empty when there were less than 5 values.
            by_factor = df.groupby('factor')
            means = by_factor.iterations.transform("mean")
            stds = by_factor.iterations.transform("std")
            df = df[df.iterations.between(means - stds*3, means + stds*3)]
        method_data.append(df)
    return method_data


REPETITIONS = 100
LOW = 1
HIGH = 5

params_smaller = {
    "DIMENSION" : 100,
    "GS_SUM_RANGE" : range(10, 1001),
    "GS_MATRIX_RANGE" : range(10, 1001),
    "GS_BOXPLOT_RANGE" : range(10, 30, 2),
    "GS_SCATTERPLOT_RANGE" : range(10, 1001),
    "GS_LINE_PLOT_RANGE" : range(10, 30),

    "JACOBI_SUM_RANGE" : range(120, 1001),
    "JACOBI_MATRIX_RANGE" : range(120, 1001),
    "JACOBI_BOXPLOT_RANGE" : range(120, 241, 15),
    "JACOBI_SCATTERPLOT_RANGE" : range(120, 1001),
    "JACOBI_LINE_PLOT_RANGE" : range(120, 146),

    "JACOBI_VS_GS_LINE_PLOT_RANGE" : range(120, 241, 5)
}

params_larger = {
    "DIMENSION" : 200,
    "GS_SUM_RANGE" : range(20, 2001),
    "GS_MATRIX_RANGE" : range(20, 2001),
    "GS_BOXPLOT_RANGE" : range(20, 60, 4),
    "GS_SCATTERPLOT_RANGE" : range(20, 2001),
    "GS_LINE_PLOT_RANGE" : range(20, 62, 2),

    "JACOBI_SUM_RANGE" : range(280, 1001),
    "JACOBI_MATRIX_RANGE" : range(280, 1001),
    "JACOBI_BOXPLOT_RANGE" : range(280, 601, 40),
    "JACOBI_SCATTERPLOT_RANGE" : range(280, 1001),
    "JACOBI_LINE_PLOT_RANGE" : range(280, 306),

    "JACOBI_VS_GS_LINE_PLOT_RANGE" : range(280, 596, 15)
}

for p in [params_smaller, params_larger]:
    d = p["DIMENSION"]
    
    # === GAUSS-SEIDEL === #
    # measure_iterations_growing_diagonal(
    #     "gauss_seidel_sum_method", d, REPETITIONS, LOW, HIGH,
    #     p["GS_SUM_RANGE"])

    # measure_iterations_growing_diagonal(
    #     "gauss_seidel_matrix", d, REPETITIONS, LOW, HIGH,
    #     p["GS_MATRIX_RANGE"])

    box_plot_iterations(["gauss_seidel_sum_method", "gauss_seidel_matrix"],
                        p["GS_BOXPLOT_RANGE"], d, "box_gs_iterations", "log")

    line_plot_iterations(["gauss_seidel_sum_method", "gauss_seidel_matrix"],
                         p["GS_LINE_PLOT_RANGE"], d, "line_gs_iterations_no_outliers",
                         remove_outliers=True)
    
    plot_spectral_radius_iterations(["gauss_seidel_sum_method", "gauss_seidel_matrix"],
                                    p["GS_SCATTERPLOT_RANGE"], d,
                                    "spectral_radius_gauss_seidel", scale='linear',
                                    sample_frac=0.01)

    # === JACOBI === #
    # measure_iterations_growing_diagonal(
    #     "jacobi_sum_method", d, REPETITIONS, LOW, HIGH,
    #     p["JACOBI_SUM_RANGE"])

    # measure_iterations_growing_diagonal(
    #     "jacobi_matrix", d, REPETITIONS, LOW, HIGH,
    #     p["JACOBI_MATRIX_RANGE"])

    box_plot_iterations(["jacobi_sum_method", "jacobi_matrix"], p["JACOBI_BOXPLOT_RANGE"],
                        d, "box_jacobi_iterations", "log")

    line_plot_iterations(["jacobi_sum_method", "jacobi_matrix"], p["JACOBI_LINE_PLOT_RANGE"],
                         d, "line_jacobi_iterations_no_outliers",
                         rotate_xticklabels=True, remove_outliers=True)
    
    plot_spectral_radius_iterations(["jacobi_sum_method", "jacobi_matrix"],
                                    p["JACOBI_SCATTERPLOT_RANGE"], d,
                                    "spectral_radius_jacobi", scale='log', sample_frac=0.01)

    # === JACOBI/GS COMPARISON === #

    box_plot_iterations(
        ["jacobi_sum_method", "gauss_seidel_sum_method"], p["JACOBI_BOXPLOT_RANGE"],
        d, "box_jacobi_vs_gauss_seidel_sum_iterations", "log")

    line_plot_iterations(["jacobi_sum_method", "gauss_seidel_sum_method"],
                         p["JACOBI_VS_GS_LINE_PLOT_RANGE"], d,
                         "line_jacobi_vs_gauss_seidel_sum_iterations_no_outliers",
                         rotate_xticklabels=True, remove_outliers=True,
                         scale="log")

    box_plot_iterations(
        ["jacobi_matrix", "gauss_seidel_matrix"], p["JACOBI_BOXPLOT_RANGE"],
        d, "box_jacobi_vs_gauss_seidel_matrix_iterations", "log")

    line_plot_iterations(["jacobi_matrix", "gauss_seidel_matrix"],
                         p["JACOBI_VS_GS_LINE_PLOT_RANGE"], d,
                         "line_jacobi_vs_gauss_seidel_matrix_iterations_no_outliers",
                         rotate_xticklabels=True, remove_outliers=True,
                         scale="log")
    
    plot_spectral_radius_iterations(["jacobi_sum_method", "gauss_seidel_sum_method"],
                                    p["JACOBI_SCATTERPLOT_RANGE"], d,
                                    "spectral_radius_sum", scale='log', sample_frac=0.01)
    
    plot_spectral_radius_iterations(["jacobi_matrix", "gauss_seidel_matrix"],
                                    p["JACOBI_SCATTERPLOT_RANGE"], d,
                                    "spectral_radius_matrix", scale='log', sample_frac=0.01)
