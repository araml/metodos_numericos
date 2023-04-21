from eliminacion import *
import matplotlib.pyplot as plt
import numpy as np
import time

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

def generate_tridiagonal_vectors_test_data(dimension: int) -> (np.array, np.array, np.array):
    return (np.concatenate([ np.array([0]), np.random.rand(dimension-1) ]),
            np.random.rand(dimension),
            np.concatenate([ np.random.rand(dimension-1), np.array([0]) ]))

def vectors_to_tridiagonal_matrix(a: np.array, b: np.array, c: np.array) -> np.array:    
    return np.diag(a[1:], -1) + np.diag(b) + np.diag(c[:-1], 1)

def simulate_diffusion(dimension: int, iterations: int, alpha=1, radius=1) -> np.array:
    a, b, c = laplacian_vectors(dimension)
    negative_alpha = alpha * (-1)
    a, b, c = a * negative_alpha, b * (negative_alpha-0.5), c * negative_alpha
    
    u = np.zeros(dimension)
    for i in range(int(np.floor(dimension/2)) - radius + 1, int(np.floor(dimension/2)) + radius):
        u[i] = 1
    
    us = [u.copy()]
    for k in range(iterations):
         u_k = solve_many_tridiagonals_no_precalculation(a, b, c, np.array([u]))
         us.append(u_k.copy())
         u = u_k
    
    return np.array(us)

def mean_square_error(a1: np.array, a2: np.array) -> np.float64:
    return ((a1 - a2) ** 2).mean()

def ejercicio6(alpha=1, radius=10) -> None:
    print("Ejercicio 6.")
    diff = simulate_diffusion(dimension = 101, iterations = 1001, alpha = alpha, radius = radius)
    plt.pcolor(diff.T)
    plt.colorbar()
    plt.show()

def alpha_proportional() -> None:
    cols = 1
    print(np.linspace(1, 9, 3))
    for i in np.linspace(1, 9, 3):
        diff = simulate_diffusion(dimension = 101, iterations = 1001, alpha = i,
                                  radius = 10)
        cols = cols + 1
        plt.pcolor(diff.T)
        plt.colorbar()
        plt.show()


def measure_execution_time(function_to_measure, *args) -> float:
    start_time = time.time()
    function_to_measure(*args)
    end_time = time.time()
    return end_time-start_time


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

def test_numerical_error_no_pivoting_row_pivoting_growing_diagonal(dimension: int, exponent_range: np.array, num_test_cases: np.array, scale_x=False) -> (np.array, np.array):
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
            matrix = generate_matrix_test_data(dimension, no_pivoting=True, scale=1, diagonal_factor=factor)

            b = matrix@x
            no_pivoting_solution = solve_full_matrix(matrix, b, gaussian_elimination_no_pivoting)
            row_pivoting_solution = solve_full_matrix(matrix, b, gaussian_elimination_row_pivoting)

            no_pivoting_mse_list.append(mean_square_error(x, no_pivoting_solution))
            row_pivoting_mse_list.append(mean_square_error(x, row_pivoting_solution))

        no_pivoting_mean_square_errors.append(np.mean(no_pivoting_mse_list))
        row_pivoting_mean_square_errors.append(np.mean(row_pivoting_mse_list))
    
    return np.array(no_pivoting_mean_square_errors), np.array(row_pivoting_mean_square_errors)


def test_precalculation_vs_no_precalculation(dimension_range: np.array, num_test_cases: int, num_independent_terms: int) -> np.array:
    no_precalculation_mean_execution_times = []
    precalculation_mean_execution_times = []

    for n in dimension_range:
        no_precalculation_execution_times = []
        precalculation_execution_times = []

        for i in range(num_test_cases):
            a, b, c = generate_tridiagonal_vectors_test_data(n)
            ds = np.array([np.random.rand(n) for j in range(num_independent_terms)])
            no_precalculation_execution_times.append(measure_execution_time(solve_many_tridiagonals_no_precalculation, a, b, c, ds))
            precalculation_execution_times.append(measure_execution_time(solve_many_tridiagonals_precalculation, a, b, c, ds))

        no_precalculation_mean_execution_times.append(np.mean(no_precalculation_execution_times))
        precalculation_mean_execution_times.append(np.mean(precalculation_execution_times))
    
    return (no_precalculation_mean_execution_times, precalculation_mean_execution_times)


def compare_tridiagonal_vs_row_pivoting(dimensions, num_runs, log_progress=False) -> (np.array, np.array):
    all_row_pivoting_times = []
    all_tridiagonal_times = []

    for dim in dimensions:
        dim_row_pivoting_times = []
        dim_tridiagonal_times = []
        if log_progress:
            print(dim)
        
        a, b, c = laplacian_vectors(dim)
        d = np.random.rand(dim)
        matrix = vectors_to_tridiagonal_matrix(a, b, c)
        for i in range(num_runs):
            dim_row_pivoting_times.append(measure_execution_time(solve_full_matrix, matrix, d, gaussian_elimination_row_pivoting))
            dim_tridiagonal_times.append(measure_execution_time(solve_full_tridiagonal_matrix, matrix, d))
        
        all_row_pivoting_times.append(dim_row_pivoting_times)
        all_tridiagonal_times.append(dim_tridiagonal_times)
    
    return (all_row_pivoting_times, all_tridiagonal_times)


def plot_no_pivoting_vs_row_pivoting_numerical_error(exponent_range, no_pivoting, row_pivoting, title):
    plt.plot(exponent_range, no_pivoting, label="Sin pivoteo")
    plt.plot(exponent_range, row_pivoting, label="Con pivoteo parcial")
    plt.legend()
    plt.xlabel("Logaritmo del factor de expansión")
    plt.ylabel("Error medio cuadrático")
    plt.yscale('log')
    plt.title(title)
    plt.show()

def plot_no_pivoting_vs_row_pivoting_error_ratio(exponent_range, no_pivoting, row_pivoting, title):
    error_ratio = no_pivoting / row_pivoting
    plt.plot(exponent_range, error_ratio)
    plt.xlabel("Logaritmo del factor de expansión")
    plt.ylabel("Error sin pivoteo / error con pivoteo")
    plt.yscale('log')
    plt.title(title)
    plt.show()

def plot_precalculation_vs_no_precalculation(dimensions, no_precalculation_times, precalculation_times, title):
    plt.plot(dimensions, no_precalculation_times, label="Sin precálculo", marker='o')
    plt.plot(dimensions, precalculation_times, label="Con precálculo", marker='x')
    plt.legend()
    plt.xlabel("Dimensión")
    plt.ylabel("Tiempo de ejecución")
    plt.title(title)
    plt.show()

def plot_row_pivoting_vs_tridiagonal_time(dimensions, mean_row_pivoting_time, mean_tridiagonal_time, title, use_log_scale=False): 
    plt.plot(dimensions, mean_row_pivoting_time, '-o', label="Pivoteo parcial")
    plt.plot(dimensions, mean_tridiagonal_time, '-o', label="Tridiagonal")
    plt.legend()
    plt.xlabel("Dimensión de la matriz")
    plt.ylabel("Tiempo de ejecución (en segundos)")
    plt.title(title)
    if use_log_scale:
        plt.yscale('log')
        plt.xscale('log')
    plt.show()