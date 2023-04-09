import numpy as np
import sys

def eliminacion_gaussiana_sin_pivoteo(m: np.array) -> np.array:
    return np.eye(3)

def eliminacion_gaussiana_con_pivoteo(m: np.array) -> np.array: 
    return np.zeros((3, 3))

def eliminacion_gaussiana_tridiagonal(m: np.array) -> np.array: 
    return np.zeros((3, 3))


def gaussianElimination_noPivoting(M: np.array, b: np.array, epsilon=0):
    n = M.shape[0]

    for k in range(n):
        for i in range(k+1, n):
            if abs(M[k][k]) <= epsilon:
                raise Exception("Could not triangulate matrix due to null coefficient in diagonal")
            
            coefficient = M[i][k] / M[k][k]
            M[i] = M[i] - coefficient * M[k]
            b[i] = b[i] - coefficient * b[k] 


def gaussianElimination_rowPivoting(M: np.array, b: np.array, max=sys.float_info.max, min=sys.float_info.min):
    n = M.shape[0]

    for k in range(n):
        # choose coefficient with maximum value along the diagonal in last n-k rows
        pivotRowIndex = np.argmax(abs(M[k:,k])) + k
        M[[pivotRowIndex, k]] = M[[k, pivotRowIndex]]
        b[[pivotRowIndex, k]] = b[[k, pivotRowIndex]]
        
        for i in range(k+1, n):
            coefficient = M[i][k] / M[k][k]
            rowToSubtract = coefficient * M[k]
            solutionToSubtract = coefficient * b[k]
            # TODO: compare coefficient and M, b components with max and min
            if np.any(np.isclose(M[i], rowToSubtract)) or np.isclose(b[i], solutionToSubtract):
                print("Catastrophic cancellation risk!")
            M[i] = M[i] - rowToSubtract
            b[i] = b[i] - solutionToSubtract


def gaussianElimination_tridiagonal(M: np.array, b: np.array):
    n = M.shape[0]

    for k in range(n-1):
        coefficient = M[k+1][k] / M[k][k]
        M[k+1] = M[k+1] - coefficient * M[k]
        b[k+1] = b[k+1] - coefficient * b[k]


def solve(M, b, triangulationFunction):
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