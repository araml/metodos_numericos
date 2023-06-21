import iterative_methods
import numpy as np

def measure_n2_error(iterative_method_to_measure, m, x_0, b, iterations, eps) -> float:
    expected_result = iterative_methods.gaussian_elimination(m, b)
    actual_result = iterative_method_to_measure(m, x_0, b, iterations, eps)
    return np.linalg.norm(expected_result-actual_result, ord=2)