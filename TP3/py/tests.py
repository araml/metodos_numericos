import unittest
import numpy as np
from iterative_methods_c import *

class testPybind(unittest.TestCase):
    def test_jacobi_matrix(self):
        m = np.array([[2, 1], [1, 2]])
        x_0 = np.array([1, 1])
        b = np.array([1, 1])
        v = np.array([0.333333, 0.333333])

        result = jacobi_sum_method(m, x_0, b); 

        np.testing.assert_allclose(result, v, rtol = 1e-5)

if __name__ == '__main__':
    unittest.main()
