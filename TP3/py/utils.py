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