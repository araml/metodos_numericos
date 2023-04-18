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

def plot_laplacian() -> None:
    l1 = function_a(101)
    l2 = function_b(101)
    l3 = function_c(101)
    x = np.arange(101)

    plt.plot(x, l1, label = '(a)')
    plt.plot(x, l2, color = 'orange', label = '(b)')
    plt.plot(x, l3, color = 'green', label = '(c)')
    plt.legend()
    plt.show()

# Generate an inversible matrix with random coefficients in the range [0,scale) outside the diagonal
# and a diagonal scaled by diagonal_factor
def generate_inversible_matrix(dimension: int, scale=1, diagonal_factor=1) -> np.array:
    while True:
        matrix = np.random.rand(dimension, dimension) * scale
        scale_diagonal(matrix, diagonal_factor)
        if np.linalg.matrix_rank(matrix) == dimension:
            return matrix

# Generate a matrix that can be triangulated with no pivoting
# with coefficients in the range [0,scale) outside the diagonal
# and a diagonal scaled by diagonal_factor
def generate_no_pivoting_matrix(dimension: int, independent_term: np.array, scale=1, diagonal_factor=1, epsilon=NUMPY_EPSILON) -> np.array:
    while True:
        matrix = generate_inversible_matrix(dimension, scale, diagonal_factor)
        try:
            gaussian_elimination_no_pivoting(matrix.copy(), independent_term.copy(), epsilon)
            return matrix
        except ZeroDivisionError:
            continue

def scale_diagonal(matrix: np.array, factor: float) -> None:
    n = matrix.shape[0]
    for i in range(n):
        matrix[i, i] = matrix[i, i] * factor

def generate_matrix_test_data(dimension: int, independent_term=None, no_pivoting=False, scale=1, diagonal_factor=1, epsilon=NUMPY_EPSILON) -> np.array:
    if independent_term is not None:
        b = independent_term
    else:
        b = np.zeros(dimension)
    if no_pivoting:
        matrix = generate_no_pivoting_matrix(dimension, b, scale, diagonal_factor, epsilon)
    else:
        matrix = generate_inversible_matrix(dimension, scale, diagonal_factor)

    return matrix

def simulate_diffusion(dimension: int, iterations: int, alpha=1, radius=1) -> np.array:
    a, b, c = laplacian_vectors(dimension)
    negative_alpha = alpha * (-1)
    a, b, c = a * negative_alpha, b * 1.5 * negative_alpha, c * negative_alpha
    
    u = np.zeros(dimension)
    for i in range(int(np.floor(dimension/2)) - radius + 1, int(np.floor(dimension/2)) + radius):
        u[i] = 1
    
    us = [u.copy()]
    for k in range(iterations):
         u_k = solve_many_tridiagonals_no_precalculation(a, b, c, np.array([u]))
         us.append(u_k.copy())
         u = u_k
    
    return np.array(us)

def measure_full_matrix_numerical_error(matrix: np.array, b: np.array, error_metric, triangulation_function, *args) -> float:
    linalg_solution = np.linalg.solve(matrix, b)
    solution = solve_full_matrix(matrix, b, triangulation_function, *args)
    return error_metric(linalg_solution, solution)

def mean_square_error(a1: np.array, a2: np.array) -> np.float64:
    return ((a1 - a2) ** 2).mean()

def infinity_norm_error(a1: np.array, a2: np.array) -> np.float64:
    return np.linalg.norm(a1 - a2, np.inf)

def ejercicio6(alpha=1, radius=10) -> None:
    print("Ejercicio 6.")
    diff = simulate_diffusion(dimension = 101, iterations = 1001, alpha = alpha, radius = radius)
    plt.pcolor(diff.T)
    plt.colorbar()
    plt.show()


def test_numerical_error_no_pivoting_row_pivoting(dimension: int, exponent_range: np.array, num_test_cases: np.array, scale_x=False) -> (np.array, np.array):
    no_pivoting_mean_square_errors = []
    row_pivoting_mean_square_errors = []

    for i in exponent_range:
        no_pivoting_mse_list = []
        row_pivoting_mse_list = []
        factor = 10**float(i)

        for j in range(num_test_cases):
            x = np.array(np.random.rand(dimension))
            if scale_x:
                x = x / factor
            matrix = generate_matrix_test_data(dimension, no_pivoting=True, scale=factor)

            b = matrix@x
            no_pivoting_solution = solve_full_matrix(matrix, b, gaussian_elimination_no_pivoting)
            row_pivoting_solution = solve_full_matrix(matrix, b, gaussian_elimination_row_pivoting)

            no_pivoting_mse_list.append(mean_square_error(x, no_pivoting_solution))
            row_pivoting_mse_list.append(mean_square_error(x, row_pivoting_solution))

        no_pivoting_mean_square_errors.append(np.mean(no_pivoting_mse_list))
        row_pivoting_mean_square_errors.append(np.mean(row_pivoting_mse_list))
    
    return np.array(no_pivoting_mean_square_errors), np.array(row_pivoting_mean_square_errors)


def test_numerical_error_no_pivoting_growing_diagonal(dimension: int, exponent_range: np.array, num_test_cases: np.array, scale_x=False) -> np.array:
    no_pivoting_mean_square_errors = []

    for i in exponent_range:
        no_pivoting_mse_list = []
        row_pivoting_mse_list = []
        factor = 10**float(i)

        for j in range(num_test_cases):
            x = np.array(np.random.rand(dimension))
            if scale_x:
                x = x / factor
            matrix = generate_matrix_test_data(dimension, no_pivoting=True, scale=1, diagonal_factor=factor)

            b = matrix@x
            no_pivoting_solution = solve_full_matrix(matrix, b, gaussian_elimination_no_pivoting)
            no_pivoting_mse_list.append(mean_square_error(x, no_pivoting_solution))

        no_pivoting_mean_square_errors.append(np.mean(no_pivoting_mse_list))
    
    return np.array(no_pivoting_mean_square_errors)