# TODO(comment out if your cpu doesn't crash :)'
import os
os.environ["OMP_NUM_THREADS"] = "6" 
os.environ["OPENBLAS_NUM_THREADS"] = "6" 
os.environ["MKL_NUM_THREADS"] = "6" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" 
os.environ["NUMEXPR_NUM_THREADS"] = "6" 

import numpy as np
import matplotlib.pyplot as plt
from utils import *
from iterative_methods import *
import sys

def measure_n2_error(iterative_method_to_measure, m, x_0, b, iterations, eps) -> float:
    expected_result = iterative_methods.gaussian_elimination(m, b)
    actual_result, _ = iterative_method_to_measure(m, x_0, b, iterations, eps)
    return np.linalg.norm(expected_result-actual_result, ord=2)

def run_and_append(fn, m: np.array, b: np.array, x: np.array, 
                   acc: list, iterations: int = 4000, debug: bool = False) -> (np.array, int):
    v, iters = fn(m, b, x, iterations = iterations, debug = debug)
    acc.append(np.linalg.norm(m@v - b))
    return acc, iters

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
        jm, it1 = run_and_append(jacobi_matrix, m, b, x, jm, iterations, True)
        js, it2 = run_and_append(jacobi_sum_method, m, b, x, js, iterations, True)
        gsm, it3 = run_and_append(gauss_seidel_matrix, m, b, x, gsm, iterations, True) 
        gss, it4 = run_and_append(gauss_seidel_sum_method, m, b, x, gss, iterations, True)

        # We want to run until the algorithm stops because of the epsilon
        if max([it1, it2, it3, it4]) < iterations:
            break;

    xs = range(1, len(jm) + 1, 1)

    plt.plot(xs, jm, label=f'Jacobi matriz')       
    plt.plot(xs, js, label=f'Jacobi suma')            
    plt.plot(xs, gsm, label=f'Gauss-seidel matriz')      
    plt.plot(xs, gss, label=f'Gauss-seidel suma') 
    plt.xlabel('Iteraciones')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(f'Error over 1000 iterations') 
    plt.show()


# Crea boxplot para una dimensión específica
def run_error_boxplot(dim: int = 1000, iterations: int = 100) -> None:
    jm = []
    js = []
    gsm = []
    gss = []
    total_iters = []

    for i in range(iterations):
        sys.stdout.write(f'\rIteration {i} of {iterations}')  
        sys.stdout.flush()                                                   

        m, _, b = create_diagonally_dominant_matrix(dim)
        x = np.random.randint(low = 0, high = 100, size = dim)

        jm, it1 = run_and_append(jacobi_matrix, m, b, x, jm)
        js, it2 = run_and_append(jacobi_sum_method, m, b, x, js)
        gsm, it3 = run_and_append(gauss_seidel_matrix, m, b, x, gsm)
        gss, it4 = run_and_append(gauss_seidel_sum_method, m, b, x, gss)
     
    fig, ax = plt.subplots()
    ax.boxplot([jm, js, gsm, gss]) 
    ax.set_xticklabels(['Jacobi matriz', 'Jacobi suma', 'Gauss-Seidel matriz',
                         'Gauss-Seidel suma'], rotation=0, fontsize=8)
    plt.legend()
    plt.savefig(f'Error metodos boxplot dim: {dim}')
    #plt.show()

def run_error_experiment(max_dim: int = 5000) -> None:
    steps = int(max_dim/10)
    jm = []
    js = []
    gsm = []
    gss = []
    total_iters = []

    xs = range(steps, max_dim, steps)
    for dim in xs:
        sys.stdout.write(f'\rIteration {dim} of {max_dim}')  
        sys.stdout.flush()                                                   

        m, _, b = create_diagonally_dominant_matrix(dim)
        print(m)
        x = np.random.randint(low = 0, high = 100, size = dim)

        jm, it1 = run_and_append(jacobi_matrix, m, b, x, jm)
        js, it2 = run_and_append(jacobi_sum_method, m, b, x, js)
        gsm, it3 = run_and_append(gauss_seidel_matrix, m, b, x, gsm)
        gss, it4 = run_and_append(gauss_seidel_sum_method, m, b, x, gss)
     

    plt.plot(xs, jm, '-o', label=f'Jacobi matriz')       
    plt.plot(xs, js, '-o', label=f'Jacobi suma')            
    plt.plot(xs, gsm, '-o', label=f'Gauss-seidel matriz')      
    plt.plot(xs, gss, '-o', label=f'Gauss-seidel suma')   
    plt.xlabel('Dimension')
    plt.ylabel('Error al corte del algoritmo')
    plt.legend()
    plt.savefig(f'Error al cambiar la dimension')
    plt.show()

if __name__ == '__main__':
    #run_error_boxplot(dim = 100)
    #run_error_boxplot(dim = 500)
    #run_error_boxplot(dim = 1000)
    #run_error_boxplot(dim = 3000)
    run_error_experiment()
    #run_error_over_iterations()
