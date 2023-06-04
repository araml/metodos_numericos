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

# Ejercicio 3 c)
def PSNR(m1: np.array, m2: np.array) -> float:
    mse = (np.square(m1 - m2)).mean()
    return 20 * np.log10(255 / np.sqrt(mse))

def normalize(m1: np.array) -> np.array:
    return m1 / np.linalg.norm(m1)

def calculate_error(images: np.array, pca, use_PCA: bool, 
                    save_images: bool = False,
                    legend: str = '') -> (np.array, np.array):
    f, p = [], []
    h, w = images[0].shape
    idx = 0
    total_images = len(images)
    for im in images:
        im_compressed = pca.transform(np.array([im]))
        # PCA flattens..
        if use_PCA: 
            im_compressed = im_compressed.reshape(h, w)
        
        if save_images: 
            plt.imshow(im_compressed)
            plt.savefig(Path(figures_path, f'{legend}_image_compressed_{idx}'))
            plt.imshow(im)
            plt.savefig(Path(figures_path, f'{legend}_image_uncompressed_{idx}'))
            if idx > 60:
                return (f, f)
        sys.stdout.write(f'\rProcessing image {idx + 1} of {total_images}')
        sys.stdout.flush()
        idx = idx + 1

        # frobenius norm by default
        f.append(np.linalg.norm(normalize(im) - normalize(im_compressed)))
        p.append(PSNR(im, im_compressed))
        time.sleep(40/1000)
    print('')
    return f, [x / max(p) for x in p]

def quality_analysis(people_inside_dataset: np.array,
                     people_outside_dataset: np.array,
                     ks: list = [10], 
                     iterations: int = 100, 
                     N_outside: int = 5,
                     use_PCA = True, 
                     save_images = False) -> None:
    training_dataset = people_inside_dataset
    pca = None
    print(training_dataset[0].shape)
    # get max number of eigenvalues for training
    h, w = people_inside_dataset[0].shape
    k = max(ks)

    if use_PCA:
        pca = PCA(k, iterations)
    else:
        k = min(92, k)
        ks = [k for k in ks if k <= 92]
        print(ks)
        pca = PCA2D(k, iterations)

    pca.fit(training_dataset)
    frobenius_error_in_dataset = []
    psnr_error_in_dataset = []
    frobenius_error_outside_dataset = []
    psnr_error_outside_dataset = []
    # im1 is inside the dataset, im2 is excluded
    print('')
    idx = 0
    for k in ks: 
        print(f'Using component index: {idx} ({k}) of {len(ks)}')
        idx = idx + 1
        pca.set_components_dimension(k)
        
        f1, p1 = calculate_error(people_inside_dataset, pca, use_PCA, 
                                 save_images, f'{k}_components_inside')
        f2, p2 = calculate_error(people_outside_dataset, pca, use_PCA,
                                 save_images, f'{k}components_outside')
        frobenius_error_in_dataset.append(f1)
        psnr_error_in_dataset.append(p1)
        frobenius_error_outside_dataset.append(f2)
        psnr_error_outside_dataset.append(p2)

    if save_images:
        return 
    #print(frobenius_error_in_dataset) 
    #print(psnr_error_in_dataset)
    #print(frobenius_error_outside_dataset)
    #print(psnr_error_outside_dataset)
    
    
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
    axes.xaxis.set_minor_locator(plt.MultipleLocator(4))

    PCA_TYPE = 'PCA'
    if not use_PCA:
        PCA_TYPE = '2DPCA'

    plt.xlabel('Componentes usadas')
    plt.ylabel('Error')
    plt.title(f'Comparación del error entre {N_inside} personas adentro y {N_outside} fuera del'
              f' dataset\n para distintas cantidades de componentes usando {PCA_TYPE}')
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

 #   people = [5, 10, 20, 40]
 #   threads = []
 #   for t in [True]:
 #       components = np.linspace(1, 600, 20, dtype = int)
 #       if not t:
 #           components = np.linspace(1, 300, 60, dtype = int)
 #       for p in people: 
 #           excluded_people = images[0: 10 * p]
 #           included_people = images[10 * p:]
 #           quality_analysis(included_people, excluded_people, components, 10, p, t)

    people = [40]
    threads = []
    for p in people: 
        excluded_people = images[0: 10 * p]
        included_people = images[10 * p:]
        quality_analysis(included_people, excluded_people, 
                         np.linspace(1, 64, 32, dtype = int), 100, p, True)
 
    people = [20]
    threads = []
    for p in people: 
        excluded_people = images[0: 10 * p]
        included_people = images[10 * p:]
        quality_analysis(included_people, excluded_people, 
                         np.linspace(347, 410, 32, dtype = int), 100, p, True)
 
    people = [10]
    threads = []
    for p in people: 
        excluded_people = images[0: 10 * p]
        included_people = images[10 * p:]
        quality_analysis(included_people, excluded_people, 
                         np.linspace(253, 316, 32, dtype = int), 100, p, True)

    people = [5]
    threads = []
    for p in people: 
        excluded_people = images[0: 10 * p]
        included_people = images[10 * p:]
        quality_analysis(included_people, excluded_people, 
                         np.linspace(316, 380, 32, dtype = int), 100, p, True)
