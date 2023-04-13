from eliminacion import *
import matplotlib.pyplot as plt

def laplacian_generate_a(dimension: int) -> np.array:
    a = np.ones(dimension)
    a[0] = 0
    return a

def laplacian_generate_c(dimension: int) -> np.array:
    c = np.ones(dimension)
    c[-1] = 0
    return c 

def laplacian_generate_b(dimension: int) -> np.array:
    b = np.empty(dimension)
    b.fill(-2)
    return b

def laplacian_vectors(dimension: int) -> (np.array, np.array, np.array):
    a = laplacian_generate_a(dimension)
    b = laplacian_generate_b(dimension)
    c = laplacian_generate_c(dimension) 
    return (a, b, c)

def four_over_n_middle(dimension: int) -> np.array:
    # pretty sure the pdf is 1-indexing so we don't need to add 1 here..
    # TODO(ask)
    d = np.zeros(dimension)
    d[int(np.floor(dimension/2))] = 4/dimension
    a, b, c = laplacian_vectors(dimension)

    u = solveVectorialTridiagonal(a, b, c, d)
    return u

four_over_n_middle(3)

def four_over(dimension: int) -> np.array:
    a, b, c = laplacian_vectors(dimension)
    d = np.array([4/(dimension ** 2) for x in range(1, dimension + 1)])
    
    u = solveVectorialTridiagonal(a, b, c, d)
    return u

def xxxx(dimension: int) -> np.array:
    a, b, c = laplacian_vectors(dimension)
    d = np.array([(-1 + (2 * x) / (dimension - 1)) * 12 / dimension**2
                  for x in range(1, dimension + 1)])
    
    u = solveVectorialTridiagonal(a, b, c, d)
    return u

def experimento_laplaciano() -> None:
    l1 = four_over_n_middle(101)
    l2 = four_over(101)
    l3 = xxxx(101)
    x = np.arange(101)

    plt.plot(x, l1, color = 'blue', label = '(a)')
    plt.plot(x, l2, color = 'orange', label = '(b)')
    plt.plot(x, l3, color = 'green', label = '(c)')
    plt.legend()
    plt.show()


experimento_laplaciano()


