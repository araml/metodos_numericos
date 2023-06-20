import iterative_methods
import numpy as np
import time

def measure_execution_time(function_to_measure, *args) -> float:
    start_time = time.time()
    function_to_measure(*args)
    end_time = time.time()
    return end_time-start_time

m = np.array([[2, 1], [1, 2]])
x_0 = np.array([1, 1])
b = np.array([1, 1])

jacobi_matrix_time = measure_execution_time(iterative_methods.jacobi_matrix, m, x_0, b, 10000, 1e-6)
jacobi_sum_method_time = measure_execution_time(iterative_methods.jacobi_sum_method, m, x_0, b, 10000, 1e-6)

print(jacobi_matrix_time)
print(jacobi_sum_method_time)