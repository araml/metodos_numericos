import numpy as np
import matplotlib.pyplot as plt
from utils import *
from iterative_methods import *

def measure_n2_error(iterative_method_to_measure, m, x_0, b, iterations, eps) -> float:
    expected_result = iterative_methods.gaussian_elimination(m, b)
    actual_result, _ = iterative_method_to_measure(m, x_0, b, iterations, eps)
    return np.linalg.norm(expected_result-actual_result, ord=2)

def run_error_experiment(max_dim: int = 1000):
    steps = int(max_dim/10)
    jm = js = gsm = gss = []
    for dim in range(steps, max_dim, steps):
        m, x, b = try_create_convergent_matrix(dim)
        v1, _ = jacobi_matrix(m, b, x)
        jm.append(abs(v1 - b))
        v2, _ = jacobi_sum_method(m, b, x)
        js.append(abs(v2 - b))
        v3, _ = gauss_seidel_matrix(m, b, x)
        gsm.append(abs(v3 - b))
        v4, _ = gauss_seidel_sum_method(m, b, x)
        gss.append(abs(v4 - b))
     

    xs = range(0, max_dim, steps)
    axes.plot(xs, jm, '-o',                           
              label=f'Jacobi matriz')       
    axes.plot(xs, js, '-o',                                
              label=f'Jacobi suma')            
    axes.plot(xs, gsm, '-o',                      
              label=f'Gauss-seidel matriz')      
    axes.plot(xs, gss, '-o',                           
              label=f'Gauss-seidel suma')       
    

if __name__ == '__main__':
    run_error_experiment()
