import csv
import iterative_methods
import numpy as np
import time
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
    for d in dimensions:
        x_0 = np.random.randint(1, 10, size=d)
        execution_times = measure_time_for_dimension(
            iterative_method, d, repetitions, low, high, x_0, *args)
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow([d] + execution_times)


# sorry for repeated code pattern but i couldn't find a way around x_0
def measure_ge_time_complexity(dimensions: list,
                               repetitions: int,
                               low: int,
                               high: int,
                               filename: str) -> None:
    for d in dimensions:
        execution_times = measure_time_for_dimension(
            iterative_methods.gaussian_elimination, d, repetitions, low, high)
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow([d] + execution_times)


def plot_time_complexity(methods: list, filename: str, scale: str = 'linear'):
    ax1 = plt.subplot()
    for method in methods:
        data = {}
        with open(f"{method}.csv", newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                dimension, measurements = row[0], row[1:]
                data[dimension] = np.mean([float(m) for m in measurements])
            dimensions = data.keys()
            plt.plot(dimensions, [data[d] for d in dimensions], '-o', label=method)
            ax1.set_xticks([str(d) for d in dimensions])
            ax1.set_xticklabels([str(d) for d in dimensions], rotation=90)
    plt.legend()
    plt.yscale(scale)
    plt.xlabel("Dimensión de la matriz")
    plt.ylabel("Tiempo de ejecución promedio (en segundos)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()


REPETITIONS = 50
DIMENSIONS = range(100, 1600, 100)
ITERATIVE_METHOD_NAMES = ["jacobi_matrix", "jacobi_sum_method",
                          "gauss_seidel_matrix", "gauss_seidel_sum_method"]

# TODO: create folders for CSVs and figures
for name in ITERATIVE_METHOD_NAMES:
    measure_iterative_time_complexity(
        iterative_methods.methods_by_name[name], DIMENSIONS,REPETITIONS,
        1, 10,f"{name}.csv",10000, 1e-17)

plot_time_complexity(ITERATIVE_METHOD_NAMES, "iterative_methods_time_complexity.png")

measure_ge_time_complexity(DIMENSIONS, REPETITIONS, 1, 10, "gaussian_elimination.csv")

plot_time_complexity(["jacobi_matrix", "gauss_seidel_matrix", "gaussian_elimination"],
                     "iterative_matrix_vs_elimination_time_complexity.png")