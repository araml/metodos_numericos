import numpy as np
import pandas as pd
import os
import seaborn as sns
from data_paths import csvs_path, figures_path
from iterative_methods import *
from matplotlib import pyplot as plt
from utils import *


def measure_convergence_ratio(iterative_method_name,
                              dimension: int,
                              repetitions: int,
                              low: int,
                              high: int,
                              diagonal_expansion_factor: int,
                              x_0 = None,
                              iterations = 10000,
                              eps = 1e-6) -> pd.DataFrame:
    dict = {"converged": [], "spectral_radius": [],
            "condition_number": []}
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
        finally:
            t = iteration_matrix(m, iterative_method_name)
            dict["converged"].append(result)
            dict["spectral_radius"].append(spectral_radius(t))
            dict["condition_number"].append(np.linalg.cond(m))
        if i % 10 == 0:
            print(f"\tFinished repetition {i}")

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
        print(f"Calculating {iterative_method_name}, d = {d}")
        r = measure_convergence_ratio(
            iterative_method_name, dimension, repetitions, low, high,
            d, x_0, iterations, eps)
        results.append(r)
    df = pd.concat(results)
    df.to_csv(full_path, sep='\t', index=False)


def bar_plot_convergence(methods,
                         factors_to_plot: list,
                         filename: str,
                         iterations: int,
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


def bar_plot_spectral_radius(method_name, factors_to_plot: list, filename: str, iterations: int, num_bins = 10):
    full_path = os.path.join(csvs_path,
        f"{method_name}_convergence_ratio_{iterations}its.csv")
    df = pd.read_csv(full_path, sep='\t')
    df = df[df["factor"].isin(factors_to_plot)]

    df = df.sort_values("spectral_radius")
    spectral_radii = df["spectral_radius"].to_list()
    n = len(spectral_radii)
    bin_size = n // num_bins
    chunks = [spectral_radii[i:i+bin_size] for i in range(0, n, bin_size)]
    bins = [c[-1] for c in chunks]
    binned = df.groupby(pd.cut(df["spectral_radius"], bins=[0]+bins, labels=bins))
    means = binned.mean(numeric_only=True)

    _, ax1 = plt.subplots(figsize=(8,4))
    g = sns.barplot(data=means, x="spectral_radius", y="converged",
                    palette="rainbow", ax=ax1)
    plt.tight_layout()
    ax1.set_xticklabels(
        ['{:.4f}'.format(float(t.get_text())) for t in ax1.get_xticklabels()])
    plt.savefig(os.path.join(figures_path, filename))
    plt.close()



def bar_plot_condition_number(method_name, factors_to_plot: list, filename: str, iterations: int, num_bins = 10):
    full_path = os.path.join(csvs_path,
        f"{method_name}_convergence_ratio_{iterations}its.csv")
    df = pd.read_csv(full_path, sep='\t')
    df = df[df["factor"].isin(factors_to_plot)]

    df = df.sort_values("condition_number")
    spectral_radii = df["condition_number"].to_list()
    n = len(spectral_radii)
    bin_size = n // num_bins
    chunks = [spectral_radii[i:i+bin_size] for i in range(0, n, bin_size)]
    bins = [c[-1] for c in chunks]
    binned = df.groupby(pd.cut(df["condition_number"], bins=[0]+bins, labels=bins))
    means = binned.mean(numeric_only=True)

    _, ax1 = plt.subplots(figsize=(8,4))
    g = sns.barplot(data=means, x="condition_number", y="converged",
                    palette="rainbow", ax=ax1)
    plt.tight_layout()
    ax1.set_xticklabels(
        ['{:.3f}'.format(float(t.get_text())) for t in ax1.get_xticklabels()])
    plt.savefig(os.path.join(figures_path, filename))
    plt.close()
    

DIMENSION = 100
REPETITIONS = 100
LOW = 1
HIGH = 5
FACTORS = range(10, 1001)
GS_PLOT_PARAMS = [
    {"its": 10,
     "factors_to_plot": range(100, 210, 10),
     "sr_sum": range(95, 150),
     "sr_matrix": range(120, 201),
     "cond_sum": range(80, 151),
     "cond_matrix": range(120, 181),
     "bins": 10},
    {"its": 100,
     "factors_to_plot": range(10, 21),
     "sr_sum": range(13, 15),
     "sr_matrix": range(15, 17),
     "cond_sum": range(11, 18),
     "cond_matrix": range(11, 18),
     "bins": 10}
]
JACOBI_PLOT_PARAMS = [
    {"its": 10,
     "factors_to_plot": range(260, 1001, 50),
     "sr_sum": range(260, 601),
     "sr_matrix": range(350, 901),
     "cond_sum": range(260, 601),
     "cond_matrix": range(400, 901),
     "bins": 10},
     {"its": 100,
     "factors_to_plot": range(260, 1001, 50),
     "sr_sum": range(125, 171),
     "sr_matrix": range(125, 171),
     "cond_sum": range(120, 171),
     "cond_matrix": range(125, 171),
     "bins": 10}
]

for p in GS_PLOT_PARAMS:
    its = p["its"]
    bins = p["bins"]
    measure_convergence_ratio_growing_diagonal(
            "gauss_seidel_sum_method", DIMENSION, REPETITIONS,
            LOW, HIGH, FACTORS, None, its)
    measure_convergence_ratio_growing_diagonal(
            "gauss_seidel_matrix", DIMENSION, REPETITIONS,
            LOW, HIGH, FACTORS, None, its)
    bar_plot_convergence(
        ["gauss_seidel_sum_method", "gauss_seidel_matrix"],
        p["factors_to_plot"],
        f"gauss_seidel_convergence_dim{DIMENSION}_{its}its.png", its)
    bar_plot_spectral_radius("gauss_seidel_sum_method", p["sr_sum"],
        f"spectral_radius_convergence_gauss_seidel_sum_method_dim{DIMENSION}_{its}its.png",
        its, num_bins=bins)
    bar_plot_spectral_radius("gauss_seidel_matrix", p["sr_matrix"],
        f"spectral_radius_convergence_gauss_seidel_matrix_dim{DIMENSION}_{its}its.png",
        its, num_bins=bins)
    bar_plot_condition_number("gauss_seidel_sum_method", p["cond_sum"],
        f"condition_number_convergence_gauss_seidel_sum_method_dim{DIMENSION}_{its}its.png",
        its, num_bins=bins)
    bar_plot_condition_number("gauss_seidel_matrix", p["cond_matrix"],
        f"condition_number_convergence_gauss_seidel_matrix_dim{DIMENSION}_{its}its.png",
        its, num_bins=bins)

for p in JACOBI_PLOT_PARAMS:
    its = p["its"]
    bins = p["bins"]
    measure_convergence_ratio_growing_diagonal(
            "jacobi_sum_method", DIMENSION, REPETITIONS,
            LOW, HIGH, FACTORS, None, its)
    measure_convergence_ratio_growing_diagonal(
            "jacobi_matrix", DIMENSION, REPETITIONS,
            LOW, HIGH, FACTORS, None, its)
    bar_plot_convergence(
        ["jacobi_sum_method", "jacobi_matrix"],
        p["factors_to_plot"],
        f"jacobi_convergence_dim{DIMENSION}_{its}its.png", its)
    bar_plot_spectral_radius("jacobi_sum_method", p["sr_sum"],
        f"spectral_radius_convergence_jacobi_sum_method_dim{DIMENSION}_{its}its.png",
        its, num_bins=bins)
    bar_plot_spectral_radius("jacobi_matrix", p["sr_matrix"],
        f"spectral_radius_convergence_jacobi_matrix_dim{DIMENSION}_{its}its.png",
        its, num_bins=bins)
    bar_plot_condition_number("jacobi_sum_method", p["cond_sum"],
        f"condition_number_convergence_jacobi_sum_method_dim{DIMENSION}_{its}its.png",
        its, num_bins=bins)
    bar_plot_condition_number("jacobi_matrix", p["cond_matrix"],
        f"condition_number_convergence_jacobi_matrix_dim{DIMENSION}_{its}its.png",
        its, num_bins=bins)