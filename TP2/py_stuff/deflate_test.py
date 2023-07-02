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
# an x_0 that is normal to the first eigenvector 
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

    e, v = np.linalg.eig(m)
    print(e, '\n')
    print(v)


def test_zero_eigenvalues():
    pass

if __name__ == '__main__': 
  #  test_same_eigenvalues()
    test_normal_eigenvector()
