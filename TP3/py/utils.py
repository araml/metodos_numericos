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

# The idea is to create a diagonal matrix D whose entries are all < 1 and then 
# try to create another matrix P that is inversible, because we know that 
# similar matrix have the same eigenvalues we can create a new matrix (That is
# not only diagonal nor DDM) P * M * P^-1
# Note: the resulting matrix might not be invertible, although it most likely is
# going to be, in fact we could force it to be singular by adding a zeroe'd eigenvalue.
def try_create_convergent_matrix(dimension: int, 
                                low: float = 0,
                                high: float = 1000, 
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
