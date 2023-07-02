import numpy as np
import deflate as d


def create_matrix_with_eigenvals(dimension: int, 
                                 diagonal: np.array,
                                 low: float = 0,
                                 high: float = 100, 
                                 seed: float = None) -> (np.array, np.array, np.array):

    if seed:
        np.random.seed(seed)

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
    
    print(P)
    M = P_inv@D@P
    print(M)
    return M, P, P_inv

def test_same_eigenvalues(): 
    m, _, _ = create_matrix_with_eigenvals(5, [3, 3, 3, 3, 4])
    e, v = d.deflate(m, np.ones(m.shape[0]), 100, 5, 1e-17)
    print(e, '\n')
    print(v)

    e, v = np.linalg.eig(m)
    print(e, '\n')
    print(v)

# Idea is to create a matrix with specific eigenvalues and then use
# an x_0 that is normal to one or more of the eigenvectors 
# this should make it so once we keep trying to find more eigenvectors than the
# ones that are not normal to x_0 we won't be able to since the deflation matrix 
# M - lambda v * v^t will cancel all of these x_0s 
def test_normal_eigenvector():
    matrix_dim = (3, 3)
    D = np.zeros(matrix_dim)
    P = np.eye(3, 3)
    P_inv = np.eye(3, 3) 
    diagonal = [5, 3, 2]
    for i in range(3):
        D[i, i] = diagonal[i]

    m = P@D@P_inv

    x = np.array([0, 1, 0])
    e, v = d.deflate(m, x, 100, 3, 1e-17)
    print(e, '\n')
    print(v)

    # It doesn't matter where we start we can only get a single eigenvalue 
    x = np.array([1, 0, 0])
    e, v = d.deflate(m, x, 100, 3, 1e-17)
    print(e, '\n')
    print(v)

    e, v = np.linalg.eig(m)
    print(e, '\n')
    print(v)

# Should be somewhat like above 
def test_zero_eigenvalues():
    matrix_dim = (3, 3)
    D = np.zeros(matrix_dim)
    P = np.eye(3, 3)
    P_inv = np.eye(3, 3) 
    diagonal = [0, 3, 2]
    for i in range(3):
        D[i, i] = diagonal[i]

    m = P@D@P_inv
    x = np.ones(m.shape[0])
    e, v = d.deflate(m, x, 100, 3, 1e-17)
    print(e, '\n')
    print(v)
    
    # It doesn't matter where we start we won't get any eigenvalue
    x = np.array([1, 0, 0])
    e, v = d.deflate(m, x, 100, 3, 1e-17)
    print(e, '\n')
    print(v)

    e, v = np.linalg.eig(m)
    print(e, '\n')
    print(v)

if __name__ == '__main__': 
  #  test_same_eigenvalues()
  #  test_normal_eigenvector()
    test_zero_eigenvalues()
