import unittest
import numpy as np
from iterative_methods import *

class testPybind(unittest.TestCase):
    def test_jacobi_matrix(self):
        m = np.array([[2, 1], [1, 2]])
        x_0 = np.array([1, 1])
        b = np.array([1, 1])
        v = np.array([0.333333, 0.333333])

        result, _ = jacobi_matrix(m, x_0, b); 
    
        np.testing.assert_allclose(result, v, rtol = 1e-5)

    def test_gauss_sum(self):
        m = np.array([[2, 1], [1, 2]])
        x_0 = np.array([1, 1])
        b = np.array([1, 1])
        v = np.array([0.333333, 0.333333])

        result, _ = gauss_seidel_sum_method(m, x_0, b); 
    
        np.testing.assert_allclose(result, v, rtol = 1e-5)

    def test_gauss_matrix(self):
        m = np.array([[2, 1], [1, 2]])
        x_0 = np.array([1, 1])
        b = np.array([1, 1])
        v = np.array([0.333333, 0.333333])

        result, _ = gauss_seidel_matrix(m, x_0, b); 
    
        np.testing.assert_allclose(result, v, rtol = 1e-5)

    def test_jacobi_sum(self):
        m = np.array([[2, 1], [1, 2]])
        x_0 = np.array([1, 1])
        b = np.array([1, 1])
        v = np.array([0.333333, 0.333333])

        result, _ = jacobi_sum_method(m, x_0, b); 
    
        np.testing.assert_allclose(result, v, rtol = 1e-5)

    def test_run_debug(self):
        m = np.array([[2, 1], [1, 2]])
        x_0 = np.array([1, 1])
        b = np.array([1, 1])
        v = np.array([0.333333, 0.333333])

        _, iteration = jacobi_matrix(m, x_0, b, iterations = 4, debug = True); 
 
        print(iteration)
        self.assertTrue(iteration == 4)

if __name__ == '__main__':
    unittest.main()
