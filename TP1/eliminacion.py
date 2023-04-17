import numpy as np

NUMPY_MAX = np.finfo(np.float64).max
NUMPY_MIN = np.finfo(np.float64).min
NUMPY_EPSILON = np.finfo(np.float64).eps

def gaussian_elimination_no_pivoting(M: np.array, b: np.array, epsilon=NUMPY_EPSILON) -> None:
    n = M.shape[0]

    for k in range(n):
        for i in range(k+1, n):
            if abs(M[k][k]) <= epsilon:
                raise ZeroDivisionError("Could not triangulate matrix due to null coefficient in diagonal")
            
            coefficient = M[i][k] / M[k][k]
            M[i] = M[i] - coefficient * M[k]
            b[i] = b[i] - coefficient * b[k] 


def gaussian_elimination_row_pivoting(
        M: np.array, b: np.array,
        max=NUMPY_MAX, min=NUMPY_MIN, epsilon=NUMPY_EPSILON) -> None:

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

            if out_of_bounds(coefficient, max, min) or out_of_bounds(b[k], max, min) or np.any(M[k] >= max)\
                or np.any(M[k] <= min):
                print("Numerical error risk: multiplying by big absolute value!")
            rowToSubtract = coefficient * M[k]
            solutionToSubtract = coefficient * b[k]

            if any_close_differences(M[i], rowToSubtract) or (np.isclose(b[i], solutionToSubtract) and b[i] != solutionToSubtract):
                print("Catastrophic cancellation risk!")
            if any_absorption(M[i], rowToSubtract) or absorption(b[i], solutionToSubtract):
                print("Absorption risk!")
            M[i] = M[i] - rowToSubtract
            b[i] = b[i] - solutionToSubtract


# Helper functions

def out_of_bounds(n: float, max: float, min: float) -> bool:
    return n >= max or n <= min

def any_close_differences(a1: np.array, a2: np.array) -> bool:
    return np.any(np.logical_and(np.isclose(a1, a2), np.logical_not(a1 == a2)))
    
def absorption(n1: float, n2: float) -> bool:
    if abs(n1) > abs(n2):
        maxAbs, minAbs = n1, n2
    else:
        maxAbs, minAbs = n2, n1
    return maxAbs+minAbs == maxAbs and minAbs != 0

def any_absorption(a1: np.array, a2: np.array) -> bool:
    return np.any(np.logical_or(
        np.logical_and(a1+a2 == a1, np.logical_not(a2 == 0)),
        np.logical_and(a1+a2 == a2, np.logical_not(a1 == 0))))


# Triangulate a tridiagonal system where the matrix is represented with all coefficients,
# including zeros
def gaussian_elimination_tridiagonal(M: np.array, b: np.array, epsilon=NUMPY_EPSILON) -> None:
    n = M.shape[0]

    for k in range(n-1):
        if abs(M[k][k]) <= epsilon:
                print("Numerical error risk: dividing by small absolute value!")
        coefficient = M[k+1][k] / M[k][k]
        M[k+1] = M[k+1] - coefficient * M[k]
        b[k+1] = b[k+1] - coefficient * b[k]


# Triangulate a tridiagonal system where the matrix is represented as three vectors:
# a = [ 0, a2, a3, ... , an]
# b = [b1, b2, b3, ... , bn]
# c = [c1, c2, ..., cn-1, 0]
def gaussian_elimination_tridiagonal_vectors(a: np.array, b: np.array, c: np.array, d: np.array) -> None:
    n = a.size
    assert(b.size == n and c.size == n and d.size == n)

    for k in range(0, n-1):
        coefficient = a[k+1] / b[k]
        # subtract row above. c doesn't change because each element has a 0 above it
        b[k+1] -= coefficient * c[k]
        d[k+1] -= coefficient * d[k]


# Modifies b and returns coefficients needed to redefine solution vector
def gaussian_elimination_b_redefinition(a: np.array, b: np.array, c: np.array) -> np.array:
    n = a.size
    assert(b.size == n and c.size == n)

    coefficients = np.array([1])
    for k in range(0, n-1):
        coefficient = a[k+1] / b[k]
        coefficients = np.append(coefficients, coefficient)
        b[k+1] -= coefficient * c[k]

    return coefficients


# Transform independent term to match triangulated system
def gaussian_elimination_d_redefinition(d: np.array, coefficients: np.array) -> None:
    n = d.size
    assert (coefficients.size == n)
    for i in range(1, n):
        d[i] -= coefficients[i] * d[i-1]


# This will only work with triangulation functions that assume M is a matrix
# with all coefficients including zeroes, NOT a vector representation
def solve_full_matrix(M: np.array, b: np.array, triangulationFunction, *args) -> np.array:
    n = M.shape[0]
    assert(M.shape == (n, n))
    assert(b.size == n)

    M_, b_ = M.copy(), b.copy()
    triangulationFunction(M_, b_, *args)

    solution = np.array([])
    for i in range(n-1, -1, -1):
        equation = M_[i, i:]
        x_i = (b_[i] - np.inner(equation[1:], solution)) / equation[0]
        solution = np.insert(solution, 0, x_i)
    
    return solution


# For a list of independent terms, solve a tridiagonal system where the matrix is represented as three vectors:
# a = [ 0, a2, a3, ... , an]
# b = [b1, b2, b3, ... , bn]
# c = [c1, c2, ..., cn-1, 0]
def solve_many_tridiagonals_no_precalculation(a: np.array, b: np.array, c: np.array, ds: np.array) -> np.array:
    n = a.size
    assert(b.size == n and c.size == n and ds.shape[1] == n)

    solutions = []

    for d in ds:
        b_, d_ = b.copy(), d.copy()
        gaussian_elimination_tridiagonal_vectors(a, b_, c, d_)
        solutions = np.append(solutions, solve_triangulated_tridiagonal_vectors(b_, c, d_))

    return solutions


# Same as above, but precalculate transformation once and apply it to each independent term
def solve_many_tridiagonals_precalculation(a: np.array, b: np.array, c: np.array, ds: np.array) -> np.array:
    n = a.size
    assert(b.size == n and c.size == n and ds.shape[1] == n)

    b_= b.copy()
    coefficients = gaussian_elimination_b_redefinition(a, b_, c)
    solutions = np.array([])

    for d in ds:
        d_ = d.copy()
        gaussian_elimination_d_redefinition(d_, coefficients)
        solutions = np.append(solutions, solve_triangulated_tridiagonal_vectors(b_, c, d_))

    return solutions


# Helper function for the two previous ones. Calculate solution from triangulated matrix represented by b and c.
def solve_triangulated_tridiagonal_vectors(triangulated_b: np.array, c: np.array, independent_term: np.array) -> np.array:
    n = triangulated_b.size

    # triangulated matrix, last row has only b_n-1
    solution = np.array([independent_term[-1] / triangulated_b[-1]])
    for i in range(n-2, -1, -1):
        # substitute previously solved component, i.e. solution[0], and solve
        x_i = (independent_term[i] - c[i] * solution[0]) / triangulated_b[i]
        solution = np.insert(solution, 0, x_i)

    return solution
