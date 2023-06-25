import csv
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

def remove_outliers(data, low_q, high_q):
    low_cutoff, high_cutoff = np.quantile(data, (low_q, high_q))
    return [x for x in data if x >= low_cutoff and x <= high_cutoff]


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
    x = np.random.randint(low, high, size = dimension)
    return M, x, M@x
