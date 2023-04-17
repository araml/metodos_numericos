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
def generate_inversible_matrix(dimension: int) -> np.array:
    while True:
        matrix = np.random.rand(dimension, dimension)
        if np.linalg.matrix_rank(matrix) == dimension:
            return matrix

# Generate a matrix that can be triangulated with no pivoting
# with coefficients in the range [0,1)
def generate_no_pivoting_matrix(dimension: int):
    while True:
        matrix = generate_inversible_matrix(dimension)
        try:
            gaussian_elimination_no_pivoting(matrix.copy(), np.zeros(dimension))
            return matrix
        except ZeroDivisionError:
            continue

def scale_diagonal(matrix: np.array, factor: float) -> None:
    n = matrix.shape[0]
    for i in range(n):
        matrix[i, i] = matrix[i, i] * factor

def generate_matrix_test_data(dimension: int, scale=1, diagonal_factor=1, no_pivoting=False) -> np.array:
    if no_pivoting:
        matrix = generate_no_pivoting_matrix(dimension)
    else:
        matrix = generate_inversible_matrix(dimension)
    matrix = matrix * scale
    scale_diagonal(matrix, diagonal_factor)
    return matrix

experimento_laplaciano()

def simulate_diffusion(dimension: int, iterations: int, alpha=1, radius=1) -> np.array:
    a, b, c = laplacian_vectors(dimension)
    negative_alpha = alpha * (-1)
    a, b, c = a * negative_alpha, b * 1.5 * negative_alpha, c * negative_alpha
    
    # d = np.array([(-1 + (2 * x) / (dimension - 1)) * 12 / dimension**2 for x in range(1, dimension + 1)])
    u = np.zeros(dimension)
    for i in range(int(np.floor(dimension/2)) - radius + 1, int(np.floor(dimension/2)) + radius):
        u[i] = 1
    
    us = [u.copy()]
    for k in range(iterations):
         u_k = solve_many_tridiagonals_no_precalculation(a, b, c, np.array([u]))
         us.append(u_k.copy())
         u = u_k
    
    return np.array(us)

def ejercicio6() -> None:
    print("Ejercicio 6.")
    diff = simulate_diffusion(dimension = 101, iterations = 1001, alpha=1, radius=10)
    plt.pcolor(diff.T)
    plt.colorbar()
    plt.show()

ejercicio6()
