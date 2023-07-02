def run_and_append(fn, m: np.array, b: np.array, x: np.array, 
                   acc: list, iterations: int = 3000, debug: bool = False) -> (np.array, int):
    v, iters = fn(m, b, x, iterations = iterations, debug = debug)
    acc.append(np.linalg.norm(m@v - b))
    return acc, iters

def run_one(fn, ms, ls):
    for m in ms: 
        x = np.random.randint(low = 0, high = 100, size = m[0].shape[1])
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
    run_error_vary_spectral_radius_convergence_error()
