import csv
from iterative_methods import *
import numpy as np

# we don't include gaussian elimination because it's a baseline
methods_by_name = {
    "jacobi_matrix": jacobi_matrix,
    "jacobi_sum_method": jacobi_sum_method,
    "gauss_seidel_matrix": gauss_seidel_matrix,
    "gauss_seidel_sum_method": gauss_seidel_sum_method,
}

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
        m[i, i] = np.random.randint(max_row + 1, dimension * 10 + 1)
        if (m[i, i] <= max_row):
            assert(0)

    x = np.random.randint(low=low, high=high, size=dimension)
    return m, x, m@x

# Although the previous method was working this is a bit more random in 
# spectral radius wise 
def create_diagonally_dominant_matrix2(dimension: int, 
                                      low: int = 1, high: int = 5, 
                                      seed: float = None,
                                      diagonal_increase = 1,
                                      rnd = True) -> (np.array, np.array, np.array):

    if seed:
        np.random.seed(seed)

    m = np.random.uniform(low = 1000, high = 1100, size = (dimension, dimension))
    m = m * 100
    expand = 1
    if np.random.random() > 0.5:
        expand = np.random.randint(1, 100)

    extra = 0
    if np.random.random() > 0.5:
        extra = np.random.random() / np.random.randint(1, 100000000000)
    else:
        extra = np.random.random()
    
    for i in range(dimension):
        max_row = m.sum(axis = 1)[i] - m[i, i]
        lower = min(max_row, 200)

        m[i, i] = (lower + extra) * expand
        
        #if np.random.random() > 0.5:
        #    m[i, i] = m[i, i] + np.random.randint(1, 1000000000)

        if (m[i, i] <= max_row):
            m[i, i] = max_row + np.random.random()
            #assert(0)

    x = np.random.randint(low=low, high=10, size=dimension)
    return m, x, m@x

# Although the previous method was working this is a bit more random in 
# spectral radius wise 
def create_diagonally_dominant_matrix3(dimension: int, 
                                      low: int = 1, high: int = 5, 
                                      seed: float = None,
                                      diagonal_increase = 1,
                                      rnd = True) -> (np.array, np.array, np.array):

    if seed:
        np.random.seed(seed)

    m = np.random.randint(low = 1, high = 10, size = (dimension, dimension))
    if np.random.random() < 0.3:
        m = m / np.array([np.random.randint(1, 1000000)])
    elif np.random.random() < 0.6:
        m = m / np.array([np.random.randint(1, 10000)])
    else:
        m = m / np.array([np.random.randint(1, 1000)])
    m = -m
    m = m * 5
    for i in range(dimension):
        m[i, i] = 5
    

    x = np.random.randint(low=low, high=10, size=dimension)
    return m, x, m@x

def get_jacobi_matrix(m: np.array) -> np.array:
    jacobi_m = - m 
    jacobi_diagonal = np.zeros(m.shape)
    for i in range(m.shape[1]):
        jacobi_diagonal[i, i] = 1/m[i, i]
        jacobi_m[i, i] = 0
    
    jacobi_m = jacobi_diagonal@jacobi_m

    return jacobi_m

def generate_n_matrices_with_varying_spectral_radiuses(dimension: int = 50,
                                                       n = 200):
    r9 = [] # Radius 0.9
    r6 = [] # 
    r3 = [] # 
    r1 = [] # Radius 0.1 
    r01 = []
    r001 = []
    
    generated = 0
    while (len(r9) + len(r6) + len(r3) + len(r1) + len(r01) + len(r001) < n * 6):
        #m, _, b = create_diagonally_dominant_matrix2(dimension = dimension)
        m, _, b = create_diagonally_dominant_matrix3(dimension = dimension)

        jacobi_m = get_jacobi_matrix(m)
        r = spectral_radius(jacobi_m)
        #r = spectral_radius(m)

        sys.stdout.write(f'\r generated {generated} of {n * 6}, spectral {r} '
                         f'{len(r001)} {len(r01)} {len(r1)} {len(r3)} {len(r6)} '
                         f'{len(r9)} ')
        sys.stdout.flush()   
        
        if r < 0.001 and len(r001) < n:
            r001.append((m, b))
            generated = generated + 1
        if r > 0.001 and r < 0.01 and len(r01) < n:
            r01.append((m, b))
            generated = generated + 1
        elif r <= 0.1 and len(r1) < n:
            r1.append((m, b))
            generated = generated + 1
        elif r > 0.1 and r <= 0.3 and len(r3) < n:
            r3.append((m, b))
            generated = generated + 1
        elif r > 0.3 and r <= 0.6 and len(r6) < n:   
            r6.append((m, b))
            generated = generated + 1
        elif r > 0.6 and r <= 0.9 and len(r9) < n:
            r9.append((m, b))
            generated = generated + 1
    
    return (r001, r01, r1, r3, r6, r9)

