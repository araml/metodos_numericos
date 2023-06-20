import iterative_methods
import numpy as np

m = np.array([[2, 1], [1, 2]])
x_0 = np.array([1, 1])
b = np.array([1, 1])
x = iterative_methods.jacobi_matrix(m, x_0, b, 10000, 1e-6)

print(x)