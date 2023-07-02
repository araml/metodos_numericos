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
from math import log

def measure_n2_error(iterative_method_to_measure, m, x_0, b, iterations, eps) -> float:
    expected_result = iterative_methods.gaussian_elimination(m, b)
    actual_result, _ = iterative_method_to_measure(m, x_0, b, iterations, eps)
    return np.linalg.norm(expected_result-actual_result, ord=2)

def run_and_append(fn, m: np.array, b: np.array, x: np.array, 
                   acc: list, iterations: int = 3000, debug: bool = False) -> (np.array, int):
    v, iters = fn(m, b, x, iterations = iterations, debug = debug)
    acc.append(np.linalg.norm(m@v - b))
    return acc, iters

def run_and_append_with_random_matrix(fn, dim: int, acc: list, iterations: int = 3000,
                                      debug: bool = False) -> (np.array, int):
    m, _, b = create_diagonally_dominant_matrix(dim)
    x = np.random.randint(low = 0, high = 100, size = dim)
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
    plt.savefig(f'Error over 1000 iterations dim{dim}') 
    plt.clf()
    #plt.show()


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
        x = np.random.randint(low = 0, high = 10, size = dim)

        try:
            jm, it1 = run_and_append_with_random_matrix(jacobi_matrix, dim, jm)
            js, it2 = run_and_append_with_random_matrix(jacobi_sum_method, dim, js)
            gsm, it3 = run_and_append_with_random_matrix(gauss_seidel_matrix, dim, gsm)
            gss, it4 = run_and_append_with_random_matrix(gauss_seidel_sum_method, dim, gss)
        except:
            print(m)
            print(b)
            assert(0)
     
    fig, ax = plt.subplots()
    ax.boxplot([jm, js, gsm, gss]) 
    ax.set_xticklabels(['Jacobi matriz', 'Jacobi suma', 'Gauss-Seidel matriz',
                         'Gauss-Seidel suma'], rotation=0, fontsize=8)
    plt.legend()
    plt.savefig(f'Error metodos boxplot dim: {dim}')
    plt.clf()
    #plt.show()

def run_error_experiment() -> None:
    jm = []
    js = []
    gsm = []
    gss = []
    total_iters = []

    xs = [2 ** i for i in range(1, 13)] 
    for dim in xs:
        sys.stdout.write(f'\rIteration {dim} of {max(xs)}')  
        sys.stdout.flush()                                                   

        m, _, b = create_diagonally_dominant_matrix(dim)
        x = np.random.randint(low = 0, high = 100, size = dim)

        jm, it1 = run_and_append(jacobi_matrix, m, b, x, jm)        
        js, it2 = run_and_append(jacobi_sum_method, m, b, x, js)
        gsm, it3 = run_and_append(gauss_seidel_matrix, m, b, x, gsm)
        gss, it4 = run_and_append(gauss_seidel_sum_method, m, b, x, gss)
     

    fig, ax = plt.subplots()

    plt.plot(xs, jm, '-o', label=f'Jacobi matriz')       
    plt.plot(xs, js, '-o', label=f'Jacobi suma')            
    plt.plot(xs, gsm, '-o', label=f'Gauss-seidel matriz')      
    plt.plot(xs, gss, '-o', label=f'Gauss-seidel suma')   
    plt.xlabel('Dimension')
 
    xticklabels = []
    exponents = [int(log(x, 2)) for x in xs]                       
    xticklabels = [r"$2^{%d}$" % e for e in exponents]                 

    plt.xscale('log', base = 2)                                      
    ax.set_xticks(xs, labels = xticklabels)        

    plt.ylabel('Error al corte del algoritmo')
    plt.legend()
    plt.savefig(f'Error al cambiar la dimension cuadrados')
    plt.clf()
    #plt.show()

# A diferencia del de arriba que usa la misma matriz para todos aca generamos
# una matriz nueva cada vez
def run_error_experiment_with_random_matrix() -> None:
    jm = []
    js = []
    gsm = []
    gss = []
    total_iters = []

    xs = [2 ** i for i in range(1, 13)] #range(steps, max_dim, steps)
    for dim in xs:
        sys.stdout.write(f'\rIteration {dim} of {max(xs)}')  
        sys.stdout.flush()                                                   

        jm, it1 = run_and_append_with_random_matrix(jacobi_matrix, dim, jm)        
        js, it2 = run_and_append_with_random_matrix(jacobi_sum_method, dim, js)
        gsm, it3 = run_and_append_with_random_matrix(gauss_seidel_matrix, dim, gsm)
        gss, it4 = run_and_append_with_random_matrix(gauss_seidel_sum_method, dim, gss)
     

    fig, ax = plt.subplots()

    plt.plot(xs, jm, '-o', label=f'Jacobi matriz')       
    plt.plot(xs, js, '-o', label=f'Jacobi suma')            
    plt.plot(xs, gsm, '-o', label=f'Gauss-seidel matriz')      
    plt.plot(xs, gss, '-o', label=f'Gauss-seidel suma')   
    plt.xlabel('Dimension')

    # Thanks       
    xticklabels = []
    exponents = [int(log(x, 2)) for x in xs]                       
    xticklabels = [r"$2^{%d}$" % e for e in exponents]                 

    plt.xscale('log', base = 2)                                      
    ax.set_xticks(xs, labels = xticklabels)                        

    plt.ylabel('Error al corte del algoritmo')
    plt.legend()
    plt.savefig(f'Error al cambiar la dimension matrices random cuadrados')
    plt.clf()

def run_one(fn, ms, ls):
    for m in ms: 
        x = np.random.randint(low = 0, high = 100, size = 200)
        ls, _ = run_and_append(fn, m[0], m[1], x, ls)        

def run_spectral(ms, js, jm, gss, gsm):
    try: 
        js = run_one(jacobi_sum_method, ms[0], js)
    except:
        print(ms[0])
        print(ms)
    jm = run_one(jacobi_matrix, ms[1], js)
    gss = run_one(gauss_seidel_sum_method, ms[2], js)
    gsm = run_one(gauss_seidel_matrix, ms[3], js)
    
    return (js, jm, gss, gsm)

def run_error_vary_spectral_radius_convergence_error():
    m1 = generate_n_matrices_with_varying_spectral_radiuses(high = 1000)
    m2 = generate_n_matrices_with_varying_spectral_radiuses(high = 0)
    #m3 = generate_n_matrices_with_varying_spectral_radiuses()
    #m4 = generate_n_matrices_with_varying_spectral_radiuses()
    
    r1 = [m1[0], m2[0]] #, m3[0], m4[0]]
    r2 = [m1[1], m2[1]] #, m3[1], m4[1]]
    r3 = [m1[2], m2[2]] #, m3[2], m4[2]]
    r4 = [m1[3], m2[3]] #, m3[3], m4[3]]

    radiuses = [r1, r2]#, r3, r4]
    radiuses_str = ['0.1', '0.4'] #, '0.6', '0.9']
    for idx in range(0, 2):
        jm = []
        js = []
        gsm = []
        gss = []
        (jm, js, gsm, gss) = run_spectral(radiuses[idx], jm, js, gsm, gss)

        fig, ax = plt.subplots()
        ax.boxplot([jm, js, gsm, gss]) 
        ax.set_xticklabels(['Jacobi matriz', 'Jacobi suma', 'Gauss-Seidel matriz',
                             'Gauss-Seidel suma'], rotation=0, fontsize=8)
        plt.legend()
        plt.savefig(f'Error with varying radius dim: {radiuses_str[idx]}')
        plt.clf()


if __name__ == '__main__':
#    try: 
#        run_error_boxplot(dim = 128)
#    except:
#        pass 
#
#    try:
#        run_error_boxplot(dim = 512)
#    except:
#        pass 
#
#    try:
#        run_error_boxplot(dim = 2048)
#    except:
#        pass 
#
#    try:
#        run_error_boxplot(dim = 4096)
#    except:
#        pass 
#

#    run_error_experiment()
#    run_error_over_iterations()
#    run_error_experiment_with_random_matrix()

    run_error_vary_spectral_radius_convergence_error()
