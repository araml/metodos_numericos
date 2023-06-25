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
    jm = []
    js = []
    gsm = []
    gss = []
    for dim in range(steps, max_dim, steps):
        m, x, b = create_diagonally_dominant_matrix(dim)
        v1, _ = jacobi_matrix(m, b, x)
        jm.append(np.linalg.norm(v1 - b))
        v2, _ = jacobi_sum_method(m, b, x)
        js.append(np.linalg.norm(v2 - b))
        v3, _ = gauss_seidel_matrix(m, b, x)
        gsm.append(np.linalg.norm(v3 - b))
        v4, _ = gauss_seidel_sum_method(m, b, x)
        gss.append(np.linalg.norm(v4 - b))
     

    xs = range(steps, max_dim, steps)

    plt.plot(xs, jm, '-o', color = 'red',                          
              label=f'Jacobi matriz')       
    plt.plot(xs, js, '-o', color = 'green',                              
              label=f'Jacobi suma')            
    plt.plot(xs, gsm, '-o', color = 'blue',                      
              label=f'Gauss-seidel matriz')      
    plt.plot(xs, gss, '-o', color = 'yellow',                          
              label=f'Gauss-seidel suma')       
    plt.show()

if __name__ == '__main__':
    run_error_experiment()
