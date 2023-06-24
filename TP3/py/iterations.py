import csv
import numpy as np
import os
from matplotlib import pyplot as plt
from iterative_methods import *
from utils import adjacent_values, create_test_case


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
            m, _, b = create_test_case(dimension, low, high, diagonal_expansion_factor)
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
    if os.path.exists(filename):
        os.remove(filename)
    for d in diagonal_expansion_factors:
        iterations = measure_iterations(
            iterative_method, dimension, repetitions, low, high, d, *args)
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow([d] + iterations)


# reference: https://matplotlib.org/stable/gallery/statistics/customized_violin.html
def violin_plot_iterations(factors_to_plot: list,
                           csv_filename: str,
                           figure_filename: str,
                           scale: str,
                           bw = None) -> None:
    data = []
    x_values = []
    with open(csv_filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            factor, iterations = int(row[0]), row[1:]
            if factor in factors_to_plot:
                x_values.append(int(factor))
                data.append([int(it) for it in iterations])

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
    whiskers = np.array([adjacent_values(sorted_array, q1, q3)
                         for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
    ax.scatter(x_values, medians, marker='o', color='white', s=30, zorder=3)
    ax.vlines(x_values, quartile1, quartile3, linestyle='-', lw=5)
    ax.vlines(x_values, whiskers_min, whiskers_max, linestyle='-', lw=1)

    ax.set_yscale(scale)
    ax.set_xlabel("Factor de expansión de la diagonal")
    ax.set_ylabel("Cantidad de iteraciones hasta converger")
    ax.set_xticks(x_values)
    ax.violinplot(data, x_values, showmeans=False, showmedians=False, bw_method=bw)
    plt.savefig(figure_filename)
    plt.clf()


def line_plot_iterations(methods: list,
                         factors_to_plot: list,
                         figure_filename: str,
                         scale: str = 'linear',
                         rotate_xticklabels: bool = False) -> None:
    ax1 = plt.subplot()
    for method in methods:
        data = {}
        with open(f"{method}_iterations.csv", newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                factor, iterations = int(row[0]), row[1:]
                if factor in factors_to_plot:
                    data[factor] = np.mean([int(m) for m in iterations])
            factors = list(data.keys())
            plt.plot(factors, [data[f] for f in factors], '-o', label=method)
            ax1.set_xticks(factors)
            if rotate_xticklabels:
                ax1.set_xticklabels([str(f) for f in factors], rotation=90)

    plt.legend()
    plt.yscale(scale)
    plt.xlabel("Factor de expansión de la diagonal")
    plt.ylabel("Cantidad promedio de iteraciones hasta converger")
    plt.tight_layout()
    plt.savefig(figure_filename)
    plt.clf()


REPETITIONS = 100
DIMENSION = 100
LOW = 1
HIGH = 5
GS_SUM_RANGE = range(10, 1001)
GS_MATRIX_RANGE = range(10, 1001)
JACOBI_SUM_RANGE = range(120, 241)
JACOBI_MATRIX_RANGE = range(120, 241)

GS_VIOLIN_RANGES = [range(15, 21), range(20, 26), range(50, 56), range(100, 106)]

# === GAUSS-SEIDEL SUM === #
measure_iterations_growing_diagonal(
    gauss_seidel_sum_method, DIMENSION, REPETITIONS, LOW, HIGH,
    GS_SUM_RANGE, "gs_sum_iterations.csv")

violin_plot_iterations(
    range(10, 16), f"gs_sum_iterations.csv", f"gs_sum_iterations_10_15.png", "log")
for rg in GS_VIOLIN_RANGES:
    violin_plot_iterations(
        rg, f"gs_sum_iterations.csv", f"gs_sum_iterations_{rg[0]}_{rg[-1]}.png", "linear")

# === GAUSS-SEIDEL MATRIX === #
measure_iterations_growing_diagonal(
    gauss_seidel_matrix, DIMENSION, REPETITIONS, LOW, HIGH,
    GS_MATRIX_RANGE, "gs_matrix_iterations.csv")

violin_plot_iterations(
    range(10, 16), f"gs_matrix_iterations.csv", f"gs_matrix_iterations_10_15.png", "log")
for rg in GS_VIOLIN_RANGES:
    violin_plot_iterations(
        rg, f"gs_matrix_iterations.csv", f"gs_matrix_iterations_{rg[0]}_{rg[-1]}.png", "linear")
    
line_plot_iterations(["gs_matrix", "gs_sum"], range(10, 26), "gs_iterations.png")

# === JACOBI SUM === #
measure_iterations_growing_diagonal(
    jacobi_sum_method, DIMENSION, REPETITIONS, LOW, HIGH,
    JACOBI_SUM_RANGE, "jacobi_sum_iterations.csv")

violin_plot_iterations(
    range(120, 126), f"jacobi_sum_iterations.csv",
    f"jacobi_sum_iterations_120_125.png", "linear")

violin_plot_iterations(
    range(120, 145, 5), f"jacobi_sum_iterations.csv",
    f"jacobi_sum_iterations_120_140.png", "linear", 10000)

# == JACOBI MATRIX == #
measure_iterations_growing_diagonal(
    jacobi_matrix, DIMENSION, REPETITIONS, LOW, HIGH,
    JACOBI_MATRIX_RANGE, "jacobi_matrix_iterations.csv")

violin_plot_iterations(
    range(120, 126), f"jacobi_matrix_iterations.csv",
    f"jacobi_matrix_iterations_120_125.png", "linear")

violin_plot_iterations(
    range(120, 145, 5), f"jacobi_matrix_iterations.csv",
    f"jacobi_matrix_iterations_120_140.png", "linear", 10000)

line_plot_iterations(range(120, 146), "jacobi_sum_iterations.csv",
                     "jacobi_matrix_iterations.png", rotate_xticklabels=True)
