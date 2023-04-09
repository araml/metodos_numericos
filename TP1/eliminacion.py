import numpy as np 

def eliminacion_gaussiana_sin_pivoteo(m: np.array) -> np.array:
    return np.eye(3)

def eliminacion_gaussiana_con_pivoteo(m: np.array) -> np.array: 
    return np.zeros((3, 3))

def eliminacion_gaussiana_tridiagonal(m: np.array) -> np.array: 
    return np.zeros((3, 3))


def gaussianElimination_noPivoting(M, b):
    n = M.shape[0]

    for k in range(n):
        for i in range(k+1, n):
            if M[k][k] == 0:
                raise Exception("Could not triangulate matrix due to null coefficient in diagonal")
            
            coefficient = M[i][k] / M[k][k]
            M[i] = M[i] - coefficient * M[k]
            b[i] = b[i] - coefficient * b[k] 


def gaussianElimination_rowPivoting(M, b):
    n = M.shape[0]

    for k in range(n):
        # choose coefficient with maximum value along the diagonal in last n-k rows
        pivotRowIndex = np.argmax(abs(M[k:,k])) + k
        M[[pivotRowIndex, k]] = M[[k, pivotRowIndex]]
        b[[pivotRowIndex, k]] = b[[k, pivotRowIndex]]
        
        for i in range(k+1, n):
            coefficient = M[i][k] / M[k][k]
            M[i] = M[i] - coefficient * M[k]
            b[i] = b[i] - coefficient * b[k] 


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