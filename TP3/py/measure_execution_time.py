import iterative_methods
import numpy as np
import pandas as pd
import os
import seaborn as sns
import time
from data_paths import csvs_path, figures_path
from matplotlib import pyplot as plt
from utils import create_test_case


def measure_execution_time(function_to_measure, *args) -> float:
    start_time = time.time()
    function_to_measure(*args)
    end_time = time.time()
    return end_time-start_time


def measure_time_for_dimension(function_to_measure,
                               dimension: int,
                               repetitions: int,
                               low: int,
                               high: int,
                               *args) -> list:
    execution_times = []
    while len(execution_times) < repetitions:
        try:
            m, _, b = create_test_case(dimension, low, high, dimension*(high+1))
            execution_time = measure_execution_time(function_to_measure, m, b, *args)
            execution_times.append(execution_time)
        except:
            continue
    return execution_times


def measure_iterative_time_complexity(iterative_method,
                                      dimensions: list,
                                      repetitions: int,
                                      low: int,
                                      high: int,
                                      filename: str,
                                      *args) -> None:
    full_path = os.path.join(csvs_path, filename)
    if os.path.exists(full_path):
        os.remove(full_path)
    dict = {"dimension": [], "time": []}
    for d in dimensions:
        x_0 = np.random.randint(1, 10, size=d)
        execution_times = measure_time_for_dimension(
            iterative_method, d, repetitions, low, high, x_0, *args)
        for e in execution_times:
            dict["dimension"].append(d)
            dict["time"].append(e)
    df = pd.DataFrame(data=dict)
    df.to_csv(full_path, sep='\t', index=False)


# sorry for repeated code pattern but i couldn't find a way around x_0
def measure_ge_time_complexity(dimensions: list,
                               repetitions: int,
                               low: int,
                               high: int,
                               filename: str) -> None:
    full_path = os.path.join(csvs_path, filename)
    if os.path.exists(full_path):
        os.remove(full_path)
    dict = {"dimension": [], "time": []}
    for d in dimensions:
        execution_times = measure_time_for_dimension(
            iterative_methods.gaussian_elimination, d, repetitions, low, high)
        for e in execution_times:
            dict["dimension"].append(d)
            dict["time"].append(e)
    df = pd.DataFrame(data=dict)
    df.to_csv(full_path, sep='\t', index=False)


def plot_time_complexity(methods: list, filename: str, scale: str = 'linear'):
    datasets = []
    for method in methods:
        df = pd.read_csv(os.path.join(csvs_path, f"{method}_time.csv"), sep='\t')
        time_means = df.groupby("dimension").mean()
        datasets.append(time_means.assign(method=method))
    
    g = sns.lineplot(pd.concat(datasets), x="dimension", y="time",
                     hue="method", marker='o')
    g.set_xticks(datasets[0].axes[0].to_list())
    g.tick_params(axis='x', rotation=90)
    plt.yscale(scale)
    plt.xlabel("Dimensión de la matriz")
    plt.ylabel("Tiempo de ejecución promedio (en segundos)")
    plt.tight_layout()

    plt.savefig(os.path.join(figures_path, filename))
    plt.clf()


REPETITIONS = 100
DIMENSIONS = range(100, 1600, 100)
ITERATIVE_METHOD_NAMES = ["jacobi_matrix", "jacobi_sum_method",
                          "gauss_seidel_matrix", "gauss_seidel_sum_method"]

for name in ITERATIVE_METHOD_NAMES:
    measure_iterative_time_complexity(
        iterative_methods.methods_by_name[name], DIMENSIONS, REPETITIONS,
        1, 10, f"{name}_time.csv", 10000, 1e-17)

plot_time_complexity(ITERATIVE_METHOD_NAMES, "iterative_methods_time_complexity.png")

measure_ge_time_complexity(DIMENSIONS, REPETITIONS, 1, 10, "gaussian_elimination_time.csv")

plot_time_complexity(["jacobi_matrix", "gauss_seidel_matrix", "gaussian_elimination"],
                     "iterative_matrix_vs_elimination_time_complexity.png")