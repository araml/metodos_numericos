# TODO(comment out if your cpu doesn't crash :)'
import os
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "4" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" 
os.environ["NUMEXPR_NUM_THREADS"] = "4" 

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from iterative_methods import *
import sys

def measure_n2_error(iterative_method_to_measure, m, x_0, b, iterations, eps) -> float:
    expected_result = iterative_methods.gaussian_elimination(m, b)
    actual_result, _ = iterative_method_to_measure(m, x_0, b, iterations, eps)
    return np.linalg.norm(expected_result-actual_result, ord=2)

def run_error_over_iterations(dim: int = 900): 
    jm = []
    js = []
    gsm = []
    gss = []

    xs = range(1, 1000, 1)
    for iterations in xs:
        sys.stdout.write(f'\rIteration {iterations + 1} of {1000}')  
        sys.stdout.flush()                                                   

        m, _, b = create_diagonally_dominant_matrix(dim)
        x = np.random.randint(low = 0, high = 100, size = dim)
        v1, iters1 = jacobi_matrix(m, b, x, iterations = iterations, 
                                   debug = True)
        jm.append(np.linalg.norm(m@v1 - b))
        v2, iters2 = jacobi_sum_method(m, b, x, iterations = iterations,
                                       debug = True) 
        js.append(np.linalg.norm(m@v2 - b))
        v3, iters3 = gauss_seidel_matrix(m, b, x, iterations = iterations,
                                         debug = True)
        gsm.append(np.linalg.norm(m@v3 - b))
        v4, iters4 = gauss_seidel_sum_method(m, b, x, iterations = iterations,
                                             debug = True)   
        gss.append(np.linalg.norm(m@v4 - b))

        # We want to run until the algorithm stops because of the epsilon
        if max([iters1, iters2, iters3, iters4]) < iterations:
            break;

    xs = range(1, len(jm) + 1, 1)

    plt.plot(xs, jm, '-o', color = 'red',                          
              label=f'Jacobi matriz')       
    plt.plot(xs, js, '-o', color = 'green',                              
              label=f'Jacobi suma')            
    plt.plot(xs, gsm, '-o', color = 'blue',                      
              label=f'Gauss-seidel matriz')      
    plt.plot(xs, gss, '-o', color = 'yellow',                          
              label=f'Gauss-seidel suma') 
    plt.xlabel('Iteraciones')
    plt.ylabel('Error')
    plt.show()


def run_error_experiment(max_dim: int = 900):
    steps = int(max_dim/10)
    jm = []
    js = []
    gsm = []
    gss = []

    xs = range(steps, max_dim, steps)
    for dim in xs:
        m, _, b = create_diagonally_dominant_matrix(dim)
        x = np.random.randint(low = 0, high = 100, size = dim)
        v1, iters = jacobi_matrix(m, b, x)
        jm.append(np.linalg.norm(m@v1 - b))
        v2, _ = jacobi_sum_method(m, b, x)
        js.append(np.linalg.norm(m@v2 - b))
        v3, _ = gauss_seidel_matrix(m, b, x)
        gsm.append(np.linalg.norm(m@v3 - b))
        v4, _ = gauss_seidel_sum_method(m, b, x)
        gss.append(np.linalg.norm(m@v4 - b))
     


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
    #run_error_experiment()
    run_error_over_iterations()
