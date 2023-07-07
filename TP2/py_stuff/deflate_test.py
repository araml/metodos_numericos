import numpy as np
import deflate as d

def print_line() -> None:
    print('------------------------------------------------------------------\n')

def print_numpy(m: np.array) -> None:
    print('Running numpy\'s eigenfunction')
    e, v = np.linalg.eig(m)
    print('Eigenvalues', e, '\n')
    print('Eigenvectors\n', v)


def create_diagonal_matrix(diag: np.array) -> np.array:
    matrix_dim = (len(diag), len(diag))
    D = np.zeros(matrix_dim)
    P = np.eye(len(diag), len(diag))
    P_inv = np.eye(len(diag), len(diag))
    for i in range(len(diag)):
        D[i, i] = diag[i]

    return P@D@P_inv


def test_power_iteration_normal():
    print('Vector normal to the first eigenvalue')

    m = create_diagonal_matrix([3, 3, 2])

    print('\nMatrix: \n', m, '\n')
    print('Running our power method\n')

    x = np.array([0, 1, 1])
    e, v, i = d.power_iteration(m, x, 1000, 1e-17)
    print("Eigenvalues", e) 
    print("Eigenvectors", v, '\n')

    print_numpy(m)
    print_line()


def test_same_eigenvalue_components():
    print('Initial vector has coordinates for two of the \'autovectors\'')

    m = create_diagonal_matrix([3, 3, 2])

    print('\nMatrix: \n', m, '\n')
    print('Running our power method\n')

    x = np.array([1, 1, 0])
    e, v, i = d.power_iteration(m, x, 1000, 1e-17)
    print("Eigenvalues", e) 
    print("Eigenvectors", v, '\n')

    print_numpy(m)
    print_line()

def create_matrix_with_eigenvals(diagonal: np.array,
                                 low: float = 0,
                                 high: float = 100, 
                                 seed: float = None) -> (np.array, np.array, np.array):

    if seed:
        np.random.seed(seed)
    dimension = len(diagonal)

    matrix_dim = (dimension, dimension)
    D = np.zeros(matrix_dim)
    P = np.zeros(matrix_dim)
    P_inv = np.zeros(matrix_dim) 
    for i in range(dimension):
        D[i, i] = diagonal[i]

    # Probability of a singular matrix is 0 so we should find an invertible one
    # fairly fast.
    while True:
        P = np.random.randint(low = low, high = high, size = matrix_dim)
        try: 
            P_inv = np.linalg.inv(P)
            break
        except: 
            continue
    
    M = P_inv@D@P
    return M, P, P_inv

def test_same_eigenvalues(): 
    print('Matrix with more than one eigenvector associated to the same eigenvalue')
    m = create_diagonal_matrix([3, 3, 1])

    print('\nMatrix: \n', m, '\n')
    print('Running our deflate\n')

    e, v = d.deflate(m, np.ones(m.shape[0]), 100, 3, 1e-17)
    print("Eigenvalues", e) 
    print("Eigenvectors", v, '\n')
    
    print_numpy(m)
    print_line()

def test_same_eigenvalues(): 
    print('Matrix with more than one eigenvector associated to the same eigenvalue')
    m = create_diagonal_matrix([3, 3, 1])

    print('\nMatrix: \n', m, '\n')
    print('Running our deflate\n')

    e, v = d.deflate(m, np.ones(m.shape[0]), 100, 3, 1e-38)
    print("Eigenvalues", e) 
    print("Eigenvectors", v, '\n')
    
    print_numpy(m)
    print_line()


# Since the delfation algorithm "cancels" the eigenvalues we've already gotten
# From the matrix when trying to find the eigenvectors for the 0s we might 
# not get valid eigenvectors (ie they don't form a base)
def test_zero_eigenvalues():
    print('Matrix with 0 eigenvalues')
    m = create_diagonal_matrix([3, 1, 0])

    print('\nMatrix: \n', m, '\n')
    print('Running our deflate\n')

    e, v = d.deflate(m, np.ones(m.shape[0]), 100, 3, 1e-17)
    print("Eigenvalues", e) 
    print("Eigenvectors", v, '\n')
    
    print_numpy(m)
    print_line()

def test_zero_eigenvalues_better_epsilon():
    print('Matrix with 0 eigenvalues, better epsilon')
    m = create_diagonal_matrix([3, 1, 0])

    print('\nMatrix: \n', m, '\n')
    print('Running our deflate\n')

    e, v = d.deflate(m, np.ones(m.shape[0]), 100, 3, 1e-250)
    print("Eigenvalues", e) 
    print("Eigenvectors", v, '\n')
    
    print_numpy(m)
    print_line()

if __name__ == '__main__': 
    test_power_iteration_normal()
    test_same_eigenvalue_components()
    test_same_eigenvalues()
    test_zero_eigenvalues()
    test_zero_eigenvalues_better_epsilon()
    test_same_eigenvalue_components_get_eigenbase()
