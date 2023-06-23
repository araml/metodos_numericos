import iterative_methods_c
import numpy as np

def gaussian_elimination(m: np.array, b: np.array) -> np.array:
    return iterative_methods_c.gaussian_elimination(m, b)

def jacobi_matrix(m: np.array,
                  b: np.array,
                  x_0: np.array,
                  iterations: int = 10000,
                  eps: float = 1e-6) -> (np.array, int):
    return iterative_methods_c.jacobi_matrix(m, b, x_0, iterations, eps)

def jacobi_sum_method(m: np.array,
                      b: np.array,
                      x_0: np.array,
                      iterations: int = 10000,
                      eps: float = 1e-6) -> (np.array, int):
    return iterative_methods_c.jacobi_sum_method(m, b, x_0, iterations, eps)

def gauss_seidel_matrix(m: np.array,
                        b: np.array,
                        x_0: np.array,
                        iterations: int = 10000,
                        eps: float = 1e-6) -> (np.array, int):
    return iterative_methods_c.gauss_seidel_matrix(m, b, x_0, iterations, eps)

def gauss_seidel_sum_method(m: np.array,
                            b: np.array,
                            x_0: np.array,
                            iterations: int = 10000,
                            eps: float = 1e-6) -> (np.array, int):
    return iterative_methods_c.gauss_seidel_sum_method(m, b, x_0, iterations, eps)

methods_by_name = {
    "gaussian_elimination": gaussian_elimination,
    "jacobi_matrix": jacobi_matrix,
    "jacobi_sum_method": jacobi_sum_method,
    "gauss_seidel_matrix": gauss_seidel_matrix,
    "gauss_seidel_sum_method": gauss_seidel_sum_method,
}