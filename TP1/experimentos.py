from eliminacion import *
import matplotlib.pyplot as plt
import numpy as np

def laplacian_generate_a(dimension: int) -> np.array:
    return np.concatenate([np.array([0]), np.ones(dimension-1)])

def laplacian_generate_c(dimension: int) -> np.array:
    return np.concatenate([np.ones(dimension-1), np.array([0])])

def laplacian_generate_b(dimension: int) -> np.array:
    return np.ones(dimension) * (-2)

def laplacian_vectors(dimension: int) -> (np.array, np.array, np.array):
    a = laplacian_generate_a(dimension)
    b = laplacian_generate_b(dimension)
    c = laplacian_generate_c(dimension) 
    return (a, b, c)

def function_a(dimension: int) -> np.array:
    a, b, c = laplacian_vectors(dimension)
    d = np.zeros(dimension)
    d[int(np.floor(dimension/2))] = 4/dimension

    u = solve_many_tridiagonals_precalculation(a, b, c, np.array([d]))
    return u.flatten()

def function_b(dimension: int) -> np.array:
    a, b, c = laplacian_vectors(dimension)
    d = np.array([4/(dimension ** 2) for x in range(0, dimension)])
    u = solve_many_tridiagonals_precalculation(a, b, c, np.array([d]))
    return u.flatten()

def function_c(dimension: int) -> np.array:
    a, b, c = laplacian_vectors(dimension)
    d = np.array([(-1 + (2 * x) / (dimension - 1)) * 12 / dimension**2
                  for x in range(1, dimension + 1)])
    
    u = solve_many_tridiagonals_precalculation(a, b, c, np.array([d]))
    return u.flatten()

def experimento_laplaciano() -> None:
    l1 = function_a(101)
    l2 = function_b(101)
    l3 = function_c(101)
    x = np.arange(101)

    plt.plot(x, l1, label = '(a)')
    plt.plot(x, l2, color = 'orange', label = '(b)')
    plt.plot(x, l3, color = 'green', label = '(c)')
    plt.legend()
    plt.show()

# Generate an inversible matrix with random coefficients in the range [0,1)
def generate_inversible_matrix(dimension):
    while True:
        matrix = np.random.rand(dimension, dimension)
        if np.linalg.matrix_rank(matrix) == dimension:
            return matrix

# Generate a matrix that can be triangulated with no pivoting
# with coefficients in the range [0,1)
def generate_no_pivoting_matrix(dimension):
    while True:
        matrix = generate_inversible_matrix(dimension)
        try:
            gaussian_elimination_no_pivoting(matrix.copy(), np.zeros(dimension))
            return matrix
        except ZeroDivisionError:
            continue

def scale_diagonal(matrix, factor):
    n = matrix.shape[0]
    for i in range(n):
        matrix[i, i] = matrix[i, i] * factor

experimento_laplaciano()


