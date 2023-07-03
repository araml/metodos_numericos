# TODO(comment out if your cpu doesn't crash :)'
import os
os.environ["OMP_NUM_THREADS"] = "6" 
os.environ["OPENBLAS_NUM_THREADS"] = "6" 
os.environ["MKL_NUM_THREADS"] = "6" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "6" 
os.environ["NUMEXPR_NUM_THREADS"] = "6" 

import numpy as np 
from utils import *
import matplotlib.pyplot as plt

def run_and_append(fn, m: np.array, b: np.array, x: np.array, 
                   acc: list, iterations: int = 3000, debug: bool = False) -> (np.array, int):
    v, iters = fn(m, b, x, iterations = iterations, debug = debug)
    acc.append(np.linalg.norm(m@v - b))
    return acc, iters

def run_and_append_x_error_with_random_matrix(fn, dim: int, acc: list, iterations: int = 3000,
                                              debug: bool = False, eps = 1e-6) -> (np.array, int):
    m, orig_x, b = create_diagonally_dominant_matrix(dim)
    x = np.random.randint(low = 0, high = 100, size = dim)
    v, iters = fn(m, b, x, iterations = iterations, debug = debug)
    acc.append(np.linalg.norm(v - orig_x))
    return acc, iters

def run_one(fn, ms, ls):
    for m in ms: 
        x = np.random.randint(low = 0, high = 100, size = m[0].shape[1])
        ls, _ = run_and_append(fn, m[0], m[1], x, ls)        

    return ls

def run_one_x_error(fn, ms, ls):
    for m in ms: 
        x = np.random.randint(low = 0, high = 100, size = m[0].shape[1])
        ls, _ = run_and_append_x_error_with_random_matrix(fn, m[0], m[1], x, ls)        

    return ls


def run_spectral(ms, js, jm, gss, gsm):
    js = run_one(jacobi_sum_method, ms[0], js)
    jm = run_one(jacobi_matrix, ms[1], js)
    gss = run_one(gauss_seidel_sum_method, ms[2], js)
    gsm = run_one(gauss_seidel_matrix, ms[3], js)
    
    return (js, jm, gss, gsm)

def run_error_vary_spectral_radius_convergence_error(dim: int = 100, n = 200):
    m1 = generate_n_matrices_with_varying_spectral_radiuses(dim, n)
    m2 = generate_n_matrices_with_varying_spectral_radiuses(dim, n)
    m3 = generate_n_matrices_with_varying_spectral_radiuses(dim, n)
    m4 = generate_n_matrices_with_varying_spectral_radiuses(dim, n)
    
    r1 = [m1[0], m2[0], m3[0], m4[0]]
    r2 = [m1[1], m2[1], m3[1], m4[1]]
    r3 = [m1[2], m2[2], m3[2], m4[2]]
    r4 = [m1[3], m2[3], m3[3], m4[3]]
    r5 = [m1[4], m2[4], m3[4], m4[4]]
    r6 = [m1[5], m2[5], m3[5], m4[5]]

    radiuses = [r1, r2, r3, r4, r5, r6]
    radiuses_str = ['0.001', '0.01', '0.1', '0.4', '0.6', '0.9']
    radiuses_v = [0.001, 0.01, 0.1, 0.4, 0.6, 0.9]
    funcs = [jacobi_matrix, jacobi_sum_method, gauss_seidel_matrix,
                gauss_seidel_sum_method]
    funcs_names = ['jacobi_matrix', 'jacobi_sum_method', 'gauss_seidel_matrix',
                'gauss_seidel_sum_method']

    for idx in range(len(funcs)):
        ys = []
        fn = funcs[idx]
        fn_name = funcs_names[idx]
        for d in m1:
            res = []
            res = run_one(fn, d, res)
            res = np.mean(res)
            ys.append(res)
        plt.plot(radiuses_v, ys, label = fn_name, marker='o')
    
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Radio espectral')
    plt.ylabel('Error promedio')

    plt.savefig(f'plot plot for varying radiuses')
#    for idx in range(0, 6):
#        jm = []
#        js = []
#        gsm = []
#        gss = []
#        (jm, js, gsm, gss) = run_spectral(radiuses[idx], jm, js, gsm, gss)
#
#        fig, ax = plt.subplots()
#        ax.boxplot([jm, js, gsm, gss]) 
#        ax.set_xticklabels(['Jacobi matriz', 'Jacobi suma', 'Gauss-Seidel matriz',
#                             'Gauss-Seidel suma'], rotation=0, fontsize=8)
#        plt.savefig(f'Error with varying radius dim: {radiuses_str[idx]}.png')
#        plt.clf()


if __name__ == '__main__':
    run_error_vary_spectral_radius_convergence_error()
