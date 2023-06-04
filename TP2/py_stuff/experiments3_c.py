import numpy as np
import matplotlib.pyplot as plt
from data_paths import * 
from figure_generation import *
from pathlib import Path
from PCA import *
from PCA2D import PCA2D
from utilities import read_images
from parser import create_parser
from utilities import average_execution_time, centre_images
from threading import Thread
from multiprocessing import Process

SAMPLES_PER_PERSON = 10
PLOT_COLOURS = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Ejercicio 3 c)
def PSNR(m1: np.array, m2: np.array) -> float:
    mse = (np.square(m1 - m2)).mean()
    return 20 * np.log10(255 / np.sqrt(mse))

def normalize(m1: np.array) -> np.array:
    return m1 / np.linalg.norm(m1)

def calculate_error(images: np.array, pca, use_PCA: bool) -> (np.array, np.array):
    f, p = [], []
    h, w = images[0].shape
    for im in images:
        im_compressed = pca.transform(np.array([im]))
        # PCA flattens..
        if use_PCA: 
            im_compressed = im_compressed.reshape(h, w)
        # frobenius norm by default
        f.append(np.linalg.norm(normalize(im) - normalize(im_compressed)))
        p.append(PSNR(im, im_compressed))
    
    return f, [x / max(p) for x in p]

def quality_analysis(people_inside_dataset: np.array,
                     people_outside_dataset: np.array,
                     ks: list = [10], 
                     iterations: int = 100, 
                     N_outside: int = 5,
                     use_PCA = True) -> None:
    training_dataset = people_inside_dataset
    pca = None
    print(training_dataset[0].shape)
    # get max number of eigenvalues for training
    h, w = people_inside_dataset[0].shape
    ks = [k for k in ks if k <= h * w]
    k = max(ks)
    
    if use_PCA:
        pca = PCA(k, iterations)
    else:
        pca = PCA2D(k, iterations)

    pca.fit(training_dataset)
    frobenius_error_in_dataset = []
    psnr_error_in_dataset = []
    frobenius_error_outside_dataset = []
    psnr_error_outside_dataset = []
    # im1 is inside the dataset, im2 is excluded
    for k in ks: 
        pca.set_components_dimension(k)
        
        f1, p1 = calculate_error(people_inside_dataset, pca, use_PCA)
        f2, p2 = calculate_error(people_outside_dataset, pca, use_PCA)
        frobenius_error_in_dataset.append(f1)
        psnr_error_in_dataset.append(p1)
        frobenius_error_outside_dataset.append(f2)
        psnr_error_outside_dataset.append(p2)
        

    print(frobenius_error_in_dataset) 
    print(psnr_error_in_dataset)
    print(frobenius_error_outside_dataset)
    print(psnr_error_outside_dataset)
    
    
    average = lambda l: [np.mean(x) for x in l]

    frobenius_error_in_dataset = average(frobenius_error_in_dataset)
    psnr_error_in_dataset = average(psnr_error_in_dataset)
    frobenius_error_outside_dataset = average(frobenius_error_outside_dataset)
    psnr_error_outside_dataset = average(psnr_error_outside_dataset)

    _, axes = plt.subplots(figsize=(8, 6))

    N_inside = 41 - N_outside
    axes.plot(ks, frobenius_error_in_dataset, '-o',
              label=f'frobenius con {N_inside} personas en el dataset')
    axes.plot(ks, psnr_error_in_dataset, '-o', 
              label=f'PSNR con {N_inside} personas en el dataset')
    axes.plot(ks, frobenius_error_outside_dataset, '-o', 
              label=f'frobenius {N_outside} personas fuera del dataset')
    axes.plot(ks, psnr_error_outside_dataset, '-o', 
              label=f'PSNR con {N_outside} personas fuera del dataset')

    PCA_TYPE = 'PCA'
    if not use_PCA:
        PCA_TYPE = '2DPCA'

    plt.xlabel('Componentes usadas')
    plt.ylabel('Error')
    plt.title(f'Comparación del error entre {N_inside} personas adentro y {N_outside} fuera del'
              f'dataset\n para distintas cantidades de componentes usando {PCA_TYPE}')
    plt.xticks(ks)
    plt.ylim(bottom=0.0)
    plt.legend()
    file_path = Path(figures_path, f'Comparacion de error con '
                     f'{PCA_TYPE}_adentro_{N_inside}_afuera_{N_outside}')
    plt.savefig(file_path)

if __name__ == '__main__': 
    parser = create_parser("experiments")
    args = parser.parse_args()
    number_of_eigenvectors = args.number_of_eigenvectors
    similarity_2dpca = args.similarity_2dpca
    
    images = read_images(Path(faces_path), 
                         args.scale_down_factor)
    
    # Run exercise 3a
    #ejercicio_3a(PCA2D, images, 1, 2)
    
    #max_components = min(number_of_eigenvectors, images[0].size)
    #if similarity_2dpca:
    #    max_components = min(images[0].shape)

    #k_range = np.linspace(1, max_components, 10, dtype=int)
    #for its in [1, 2, 3, 4, 5, 8, 10, 15, 20]:
    #    ejercicio_3b(images, k_range, use_2d=similarity_2dpca, iterations=its)
    
    #max_components = min(images[0].shape)
    #quality_analysis(images, single_person, excluded_person)
    # Runs 2DPCA

    people = [5, 10, 20, 40]
    threads = []
    for p in people: 
        excluded_people = images[0: 9 * p]
        included_people = images[9 * p + 1:]
        th = Process(target = quality_analysis, 
                     args = (included_people, excluded_people, 
                            np.linspace(1, 200, 50, dtype = int), 100, p, True))
        threads.append(th)
                #np.linspace(1, 92, 23, dtype = int), 100, p, False) 

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # pca = PCA2D(40, filename="amogus")
    # pca.fit(images)
    # for its in [1, 2, 3, 4, 5, 8, 10, 15, 20]:
    #     ejercicio_3b(images, k_range, its)
    #ejercicio_3d_2dpca(images, k_range, 50)
    # print(images.shape)
    #quality_analysis(images, single_person, excluded_person, args)
    #quality_analysis(np.array(), images, False)
