import csv
from iterative_methods import *
import numpy as np

def create_test_case(dimension: int, low: int, high: int, diagonal_expansion_factor=None):
    m = np.random.randint(low=low, high=high, size=(dimension, dimension))
    for i in range(dimension):
        while m[i, i] == 0:
            m[i, i] = np.random.randint(low=low, high=high)
        if diagonal_expansion_factor:
            m[i, i] *= diagonal_expansion_factor
    x = np.random.randint(low=low, high=high, size=dimension)
    return (m, x, m@x)

# source: https://matplotlib.org/stable/gallery/statistics/customized_violin.html
def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def read_data_from_csv(keys_to_read: list, csv_filename: str, key_type, value_type):
    x_values = []
    data = []
    with open(csv_filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            key, values = int(row[0]), row[1:]
            if key in keys_to_read:
                x_values.append(key_type(key))
                data.append([value_type(v) for v in values])
    return x_values, data

def iteration_matrix(m: np.array, method_name: str) -> np.array:
    l = np.tril(m, -1) * (-1)
    u = np.triu(m, 1) * (-1)
    d = m + l + u
    if method_name.startswith("jacobi"):
        return np.linalg.inv(d)@(l+u)
    elif method_name.startswith("gauss_seidel"):
        return np.linalg.inv(d-l)@u

def spectral_radius(m: np.array) -> np.array:
    eigenvalues = np.linalg.eigvals(m)
    return np.abs(max(eigenvalues))

# Edit: Although this should guarantee that Ax converges to something it doesn't 
# guarantees it will under jacobi or gauss seidel
# The idea is to create a diagonal matrix D whose entries are all < 1 and then 
# try to create another matrix P that is inversible, because we know that 
# similar matrix have the same eigenvalues we can create a new matrix (That is
# not only diagonal nor DDM) P * M * P^-1
# Note: the resulting matrix might not be invertible, although it most likely is
# going to be, in fact we could force it to be singular by adding a zeroe'd eigenvalue.
def try_create_convergent_matrix(dimension: int, 
                                low: float = 0,
                                high: float = 100, 
                                seed: float = None) -> (np.array, np.array, np.array):

    if seed:
        np.random.seed(seed)

    matrix_dim = (dimension, dimension)
    D = np.zeros(matrix_dim)
    for i in range(dimension):
        D[i, i] = np.random.random()

    P = P_inv = D

    # Probability of a singular matrix is 0 so we should find an invertible one
    # fairly fast.
    while True:
        P = np.random.randint(low = low, high = high, size = matrix_dim)
        try: 
            P_inv = np.linalg.inv(P)
            break
        except: 
            continue
    
    M = P@D@P_inv
    x = np.random.randint(low=low, high=high, size=dimension)
    return M, x, M@x

import random
import sys

def create_diagonally_dominant_matrix(dimension: int, 
                                      low: float = 1, high: float = 10, 
                                      seed: float = None,
                                      diagonal_increase = 1,
                                      rnd = True) -> (np.array, np.array, np.array):

    if seed:
        np.random.seed(seed)

    m = np.random.randint(low=low, high=high, size=(dimension, dimension))

    
    for i in range(dimension):
        max_row = m.sum(axis = 1)[i] - m[i, i]
        if rnd:
            diagonal_increase = np.random.randint(1, 10)
        m[i, i] = np.random.randint(max_row + 1, dimension * 5 * diagonal_increase + 1)
        if (m[i, i] <= max_row):
            print(m)
            print("ASD", max_row, m[i, i])
            assert(0)

    x = np.random.randint(low=low, high=high, size=dimension)
    return m, x, m@x

def create_diagonally_dominant_matrix2(dimension: int, 
                                      low: float = 1, high: float = 10, 
                                      seed: float = None,
                                      diagonal_increase = 1,
                                      rnd = True) -> (np.array, np.array, np.array):

    if seed:
        np.random.seed(seed)

    m = np.random.random((dimension, dimension))
    
    for i in range(dimension):
        max_row = m.sum(axis = 1)[i] - m[i, i]
        m[i, i] = np.random.randint(min(max_row + 1, 200), max(max_row + 2, high))
        if (m[i, i] <= max_row):
            assert(0)

    x = np.random.randint(low=low, high=10, size=dimension)
    return m, x, m@x

def create_diagonally_dominant_matrix3(dimension: int, min_r, max_r, 
                                       low: float = 1, high: float = 10, 
                                       seed: float = None,
                                       diagonal_increase = 1,
                                       rnd = True) -> (np.array, np.array, np.array):

    if seed:
        np.random.seed(seed)
    
    while True:
        m = np.random.random((dimension, dimension))
        for i in range(dimension):
            m[i, i] = 0
        #print('HUH', np.random.randint(1, 100000))
        #m / np.random.randint(1, 100000)
        #print(m)
        r = spectral_radius(m)
        sys.stdout.write(f'\r spectral {r}')
        sys.stdout.flush()   
        
        if r > min_r and r < max_r:
            m = - m
            diag = np.random.randint(1, 300)
            m / np.array([diag])
            for i in range(dimension):
                m[i, i] = diag
            x = np.random.randint(low=low, high=10, size=dimension)
            return m, x, m@x, r

def generate_n_matrices_with_varying_spectral_radiuses(dimension: int = 3, n =
        5, high = 1000):
    r9 = [] # Radius 0.9
    r6 = []
    r3 = []
    r1 = [] # Radius 0.1 
    
    generated = 0
    while (len(r9) < n or len(r6) < n or len(r3) < n or len(r1) < n):
 #       m, _, b = create_diagonally_dominant_matrix2(dimension, high = high)
 #       jacobi_m = - m 
 #       jacobi_diagonal = np.zeros((dimension, dimension))
 #       for i in range(dimension):
 #           jacobi_diagonal[i, i] = 1/m[i, i]
 #           jacobi_m[i, i] = 0
 #       
 #       jacobi_m = jacobi_diagonal@jacobi_m
 #       r = spectral_radius(jacobi_m)
        
        m, _, b, r = create_diagonally_dominant_matrix3(dimension, 0, 0.3)

        sys.stdout.write(f'\r generated {generated} of 800, spectral {r}')
        sys.stdout.flush()   
        
        if len(r1) < n:
            r1.append((m, b))
            generated = generated + 1
        elif len(r3) < n:
            r3.append((m, b))
            generated = generated + 1
        elif len(r6) < n:   
            r6.append((m, b))
            generated = generated + 1
        elif len(r9) < n:
            r9.append((m, b))
            generated = generated + 1

    return (r1, r3, r6, r9)

# we don't include gaussian elimination because it's a baseline
methods_by_name = {
    "jacobi_matrix": jacobi_matrix,
    "jacobi_sum_method": jacobi_sum_method,
    "gauss_seidel_matrix": gauss_seidel_matrix,
    "gauss_seidel_sum_method": gauss_seidel_sum_method,
}
