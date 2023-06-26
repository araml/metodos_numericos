import numpy as np
import pandas as pd
import os
import seaborn as sns
from data_paths import csvs_path, figures_path
from iterative_methods import *
from matplotlib import pyplot as plt
from utils import create_test_case


def measure_convergence_ratio(iterative_method_name,
                              dimension: int,
                              repetitions: int,
                              low: int,
                              high: int,
                              diagonal_expansion_factor: int,
                              x_0 = None,
                              iterations = 10000,
                              eps = 1e-6):
    dict = {"converged": []}
    iterative_method = methods_by_name[iterative_method_name]
    if x_0 is None:
        x_0 = np.random.randint(low, high, dimension)
    for i in range(repetitions):
        result = 1
        try:
            m, _, b = create_test_case(dimension, low, high, diagonal_expansion_factor)
            _, _ = iterative_method(m, b, x_0, iterations, eps)
        except:
            result = 0 # did not converge
        dict["converged"].append(result)
    dict["factor"] = diagonal_expansion_factor
    dict["method"] = iterative_method_name
    dict["iterations"] = iterations
    dict["epsilon"] = eps
    dict["dimension"] = dimension
    return pd.DataFrame(data=dict)


def measure_convergence_ratio_growing_diagonal(iterative_method_name,
                                               dimension: int,
                                               repetitions: int,
                                               low: int,
                                               high: int,
                                               diagonal_expansion_factors: list,
                                               x_0 = None,
                                               iterations = 10000,
                                               eps = 1e-6):
    full_path = os.path.join(csvs_path,
        f"{iterative_method_name}_convergence_ratio_{iterations}its.csv")
    if os.path.exists(full_path):
        os.remove(full_path)
    results = []
    for d in diagonal_expansion_factors:
        r = measure_convergence_ratio(
            iterative_method_name, dimension, repetitions, low, high,
            d, x_0, iterations, eps)
        results.append(r)
    df = pd.concat(results)
    df.to_csv(full_path, sep='\t', index=False)


def bar_plot_convergence(methods,
                         factors_to_plot: list,
                         filename: str,
                         iterations = int,
                         scale: str = "linear"):
    method_data = []
    for method in methods:
        full_path = os.path.join(csvs_path,
            f"{method}_convergence_ratio_{iterations}its.csv")
        df = pd.read_csv(full_path, sep='\t')
        df = df[df["factor"].isin(factors_to_plot)]
        method_data.append(df)
    means = []
    for df in method_data:
        method = df["method"].iloc[0]
        mean = df.groupby("factor").mean(numeric_only=True)
        means.append(mean.assign(method=method))
    df = pd.concat(means)
    g = sns.barplot(data=df, x=df.index, y="converged", hue="method")
    plt.xlabel("Factor de expansión de la diagonal")
    plt.ylabel("Proporción de casos en los que el método converge")
    plt.tight_layout()
    plt.yscale(scale)
    plt.savefig(os.path.join(figures_path, filename))
    plt.close()
    

DIMENSION = 100
REPETITIONS = 100
LOW = 1
HIGH = 5
FACTORS = range(10, 1001)
ITERATIONS = range(10, 51, 10)
GS_BAR_PLOT_PARAMS = [
    (10, range(100, 210, 10)),
    (100, range(10, 21))]
JACOBI_BAR_PLOT_PARAMS = [
    (10, range(250, 800, 50)),
    (100, range(125, 171, 4))
]

for its, factors_to_plot in GS_BAR_PLOT_PARAMS:
    measure_convergence_ratio_growing_diagonal(
            "gauss_seidel_sum_method", DIMENSION, REPETITIONS,
            LOW, HIGH, FACTORS, None, its)
    measure_convergence_ratio_growing_diagonal(
            "gauss_seidel_matrix", DIMENSION, REPETITIONS,
            LOW, HIGH, FACTORS, None, its)
    bar_plot_convergence(
        ["gauss_seidel_sum_method", "gauss_seidel_matrix"],
        factors_to_plot,
        f"gauss_seidel_convergence_dim{DIMENSION}_{its}its.png", its)

for its, factors_to_plot in JACOBI_BAR_PLOT_PARAMS:
    measure_convergence_ratio_growing_diagonal(
            "jacobi_sum_method", DIMENSION, REPETITIONS,
            LOW, HIGH, FACTORS, None, its)
    measure_convergence_ratio_growing_diagonal(
            "jacobi_matrix", DIMENSION, REPETITIONS,
            LOW, HIGH, FACTORS, None, its)
    bar_plot_convergence(
        ["jacobi_sum_method", "jacobi_matrix"],
        factors_to_plot,
        f"jacobi_convergence_dim{DIMENSION}_{its}its.png", its)