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
        max=NUMPY_MAX, min=NUMPY_MIN, epsilon=NUMPY_EPSILON,
        turn_off_warnings = False) -> None:
    n = M.shape[0]

    for k in range(n):
        # choose coefficient with maximum value along the diagonal in last n-k rows
        pivot_row_index = find_pivot_row_index(M, k, k, n)
        M[[pivot_row_index, k]] = M[[k, pivot_row_index]]
        b[[pivot_row_index, k]] = b[[k, pivot_row_index]]
        
        for i in range(k+1, n):
            if abs(M[k][k]) <= epsilon and not turn_off_warnings:
                print("Numerical error risk: dividing by small absolute value!")
            coefficient = M[i][k] / M[k][k]

            if (out_of_bounds(coefficient, max, min) or out_of_bounds(b[k], max, min) or any_out_of_bounds(M[k], max, min)) \
                and not turn_off_warnings:
                print("Numerical error risk: multiplying by big absolute value!")
            row_to_subtract = coefficient * M[k]
            solution_to_subtract = coefficient * b[k]

            if (any_absorption(M[i], row_to_subtract) or absorption(b[i], solution_to_subtract)) and not turn_off_warnings:
                print("Absorption risk!")
            M[i] = M[i] - row_to_subtract
            b[i] = b[i] - solution_to_subtract


# Helper functions. Now 100% Numpy-free!
def find_pivot_row_index(matrix: np.array, column_index: int, start_row: int, end_row: int):
    max_element_row_index = start_row
    max_element = abs(matrix[max_element_row_index, column_index])
    
    for current_row in range(start_row, end_row):
        current_element = abs(matrix[current_row, column_index])
        if current_element > max_element:
            max_element_row_index = current_row
            max_element = current_element

    return max_element_row_index

def out_of_bounds(n: float, max: float, min: float) -> bool:
    return n >= max or n <= min

def any_out_of_bounds(a: np.array, max: float, min: float) -> bool:
    for i in range(a.size):
        if out_of_bounds(a[i], max, min):
            return True
    return False
    
def absorption(n1: float, n2: float) -> bool:
    if abs(n1) > abs(n2):
        max_abs, min_abs = n1, n2
    else:
        max_abs, min_abs = n2, n1
    return max_abs+min_abs == max_abs and min_abs != 0

def any_absorption(a1: np.array, a2: np.array) -> bool:
    for i in range(a1.size):
        if absorption(a1[i], a2[i]):
            return True
    return False

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
def solve_full_matrix(M: np.array, b: np.array, triangulation_function, *args) -> np.array:
    n = M.shape[0]
    assert(M.shape == (n, n))
    assert(b.size == n)

    M_, b_ = M.copy(), b.copy()
    triangulation_function(M_, b_, *args)

    solution = np.array([])
    for i in range(n-1, -1, -1):
        equation = M_[i, i:]
        # b_i = eq_i*x_i + ... + eq_(n-1)*x_(n-1)
        # => x_i = (b_i - eq_(i+1)*x_(i+1) - ... - eq_(n-1)*x_(n-1)) / eq_i
        x_i = (b_[i] - inner_product(equation[1:], solution)) / equation[0]
        solution = np.insert(solution, 0, x_i)
    
    return solution

def solve_full_tridiagonal_matrix(M: np.array, b: np.array, *args) -> np.array:
    n = M.shape[0]
    assert(M.shape == (n, n))
    assert(b.size == n)

    M_, b_ = M.copy(), b.copy()
    gaussian_elimination_tridiagonal(M_, b_, *args)

    solution = np.array([b_[-1] / M_[-1, -1]]) # start with bottom element
    # solve remaining unknowns with backwards substitution
    for i in range(n-2, -1, -1):
        equation = M_[i]
        x_i = (b_[i] - equation[i+1]*solution[0]) / equation[i]
        solution = np.insert(solution, 0, x_i)
        # equation = M_[i, i:i+2] # only two non-null coefficients left
        # # b_i = eq_i*x_i + ... + eq_(n-1)*x_(n-1)
        # # => x_i = (b_i - eq_(i+1)*x_(i+1) - ... - eq_(n-1)*x_(n-1)) / eq_i
        # x_i = (b_[i] - inner_product(equation[1:], solution)) / equation[0]
        # solution = np.insert(solution, 0, x_i)
    
    return solution

def inner_product(equation: np.array, solution: np.array):
    return sum([equation[i] * solution[i] for i in range(len(equation))])

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
