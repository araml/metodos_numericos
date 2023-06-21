import iterative_methods
import numpy as np
import time
from utils import create_test_case

def measure_execution_time(function_to_measure, *args) -> float:
    start_time = time.time()
    function_to_measure(*args)
    end_time = time.time()
    return end_time-start_time

def measure_time_for_dimension(function_to_measure, repetitions, dimension, *args):
    execution_times = []
    while len(execution_times) < repetitions:
        try:
            m, _, b = create_test_case(dimension, 1, 10, dimension*11)
            execution_time = measure_execution_time(function_to_measure, m, b, *args)
            execution_times.append(execution_time)
        except:
            continue
    return execution_times

DIMENSION = 1000
REPETITIONS = 50
x_0 = np.random.randint(1, 10, size=DIMENSION)
print(np.mean(measure_time_for_dimension(iterative_methods.jacobi_matrix, REPETITIONS, DIMENSION, x_0, 10000, 1e-17)))
print(np.mean(measure_time_for_dimension(iterative_methods.gaussian_elimination, REPETITIONS, DIMENSION)))