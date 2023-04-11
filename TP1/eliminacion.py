import numpy as np

NUMPY_MAX = np.finfo(np.float64).max
NUMPY_MIN = np.finfo(np.float64).min
NUMPY_EPSILON = np.finfo(np.float64).eps

def gaussianElimination_noPivoting(M: np.array, b: np.array, epsilon=0):
    n = M.shape[0]

    for k in range(n):
        for i in range(k+1, n):
            if abs(M[k][k]) <= epsilon:
                raise Exception("Could not triangulate matrix due to null coefficient in diagonal")
            
            coefficient = M[i][k] / M[k][k]
            M[i] = M[i] - coefficient * M[k]
            b[i] = b[i] - coefficient * b[k] 


def gaussianElimination_rowPivoting(
        M: np.array, b: np.array,
        max=NUMPY_MAX, min=NUMPY_MIN, epsilon=NUMPY_EPSILON, magnitudeEpsilon=NUMPY_EPSILON):

    n = M.shape[0]

    for k in range(n):
        # choose coefficient with maximum value along the diagonal in last n-k rows
        pivotRowIndex = np.argmax(abs(M[k:,k])) + k
        M[[pivotRowIndex, k]] = M[[k, pivotRowIndex]]
        b[[pivotRowIndex, k]] = b[[k, pivotRowIndex]]
        
        for i in range(k+1, n):
            if abs(M[k][k]) <= epsilon:
                print("Numerical error risk: dividing by small absolute value!")
            coefficient = M[i][k] / M[k][k]

            if outOfBounds(coefficient, max, min) or outOfBounds(b[k], max, min) or np.any(M[k] >= max) or np.any(M[k] <= min):
                print("Numerical error risk: multiplying by big absolute value!")
            rowToSubtract = coefficient * M[k]
            solutionToSubtract = coefficient * b[k]

            if np.any(np.isclose(M[i], rowToSubtract, epsilon, epsilon)) or np.isclose(b[i], solutionToSubtract, epsilon, epsilon):
                print("Catastrophic cancellation risk!")
            if differentMagnitudes(M[i], rowToSubtract, magnitudeEpsilon):
                print("Absorption risk!")
            M[i] = M[i] - rowToSubtract
            b[i] = b[i] - solutionToSubtract


def outOfBounds(n, max, min):
    return n >= max or n <= min

# TODO: change this to a better criterion
def differentMagnitudes(a1: np.array, a2: np.array, epsilon):
    t1, t2 = a1 <= epsilon, a2 <= epsilon
    return np.any(t1 != t2)


def gaussianElimination_tridiagonal(M: np.array, b: np.array):
    n = M.shape[0]

    for k in range(n-1):
        coefficient = M[k+1][k] / M[k][k]
        M[k+1] = M[k+1] - coefficient * M[k]
        b[k+1] = b[k+1] - coefficient * b[k]


def gaussianElimination_tridiagonal_vectors(a: np.array, b: np.array, c: np.array, d: np.array):
    n = a.size
    assert(b.size == n and c.size == n and d.size == n)

    for k in range(0, n-1):
        coefficient = a[k+1] / b[k]
        b[k+1] -= coefficient * c[k] # subtract row above
        d[k+1] -= coefficient * d[k]


def solve(M: np.array, b: np.array, triangulationFunction) -> np.array:
    n = M.shape[0]
    assert(M.shape == (n, n))
    assert(b.size == n)

    G, c = M.copy(), b.copy()
    triangulationFunction(G, c)

    solution = np.array([])
    for i in range(n-1, -1, -1):
        equation = G[i, i:]
        x_i = (c[i] - np.inner(equation[1:], solution)) / equation[0]
        solution = np.insert(solution, 0, x_i)
    
    return solution