# TODO(comment out if your cpu doesn't crash :)'
import os
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "4" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" 
os.environ["NUMEXPR_NUM_THREADS"] = "4" 

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from data_paths import * 
from pathlib import Path
from PCA import PCA
from PCA2D import PCA2D
from utilities import read_images


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
    #axes.xaxis.set_minor_locator(plt.MultipleLocator(10))

    PCA_TYPE = 'PCA'
    if not use_PCA:
        PCA_TYPE = '2DPCA'

    plt.xlabel('Componentes usadas')
    plt.ylabel('Error')
    plt.title(f'Comparación del error entre {N_inside} personas adentro y {N_outside} fuera del'
              f' dataset\n para distintas cantidades de componentes usando {PCA_TYPE}')
    plt.xticks(ks, rotation = 45)
    plt.ylim(bottom=0.0)
    plt.legend()
    file_path = Path(figures_path, f'Comparacion de error con '
                     f'{PCA_TYPE}_adentro_{N_inside}_afuera_{N_outside}')
    plt.savefig(file_path)


def compare_and_savefig(pca, images: np.array, ks:np.array, legend: str,
                        use_PCA: bool = False) -> None:

    idx = 0
    pca_str = '2DPCA'
    if use_PCA: 
        pca_str = 'PCA'
    for im in images:
        _, axs = plt.subplots(2, 3, figsize = (8, 6))
        for i, ax in enumerate(axs.flatten()):
            h, w = images[0].shape
            k = ks[i]
            pca.set_components_dimension(k)
            im_compressed = im
            if use_PCA:
                im_compressed = pca.transform([im])
            else:
                im_compressed = pca.transform(im)
            # PCA flattens..
            if use_PCA: 
                im_compressed = im_compressed.reshape(h, w)

            ax.imshow(im_compressed, cmap = plt.cm.gray)
            ax.set_title(f'{k} componentes')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(Path(figures_path, f'{legend}_{idx}_{pca_str}'))
        plt.clf()
        idx = idx + 1


def test_significant_features(training_dataset: np.array, p1: np.array, 
                              p2: np.array, ks: np.array, K: int,  
                              iterations = 10, use_PCA = False, 
                              legend1: str = '', legend2: str = '') -> None:
    max_k = max(ks + [K])

    if use_PCA:
        pca = PCA(max_k, iterations)
    else:
        h, w = training_dataset[0].shape
        shape = training_dataset[0]
        max_k = min([max_k, h, w])
        pca = PCA2D(max_k, iterations)

    pca.fit(training_dataset)
    
    compare_and_savefig(pca, p1, ks, legend1, use_PCA = use_PCA)
    compare_and_savefig(pca, p2, ks, legend2, use_PCA = use_PCA)


def experimento_3c() -> None:
    images = read_images(Path(faces_path))
    images_bearded = read_images(Path(faces_path + '/../caras_con_barba'))
    images_unbearded = read_images(Path(faces_path + '/../caras_sin_barba'))
    people = [5, 10, 20, 40]
 #   for t in [True, False]:
 #       components = np.linspace(1, 600, 20, dtype = int)
 #       iterations = 100
 #       if not t:
 #           components = np.linspace(1, 300, 60, dtype = int)
 #           iterations = 10
 #       for p in people: 
 #           excluded_people = images[0: 10 * p]
 #           included_people = images[10 * p:]
 #           quality_analysis(included_people, excluded_people, components, 
 #                            iterations, p, t)

    # Test finer range
    people_range = [(10, 253, 316), (5, 316, 379)] #[(20, 158, 221), 
    for N_excluded, r1, r2 in people_range: 
        components = np.linspace(r1, r2, 30, dtype = int)
        excluded_people = images[0: 10 * N_excluded]
        included_people = images[10 * N_excluded:]
        quality_analysis(included_people, excluded_people, components, 100,
                         N_excluded, use_PCA = True)
                         
 #   p1 = images[140:150]
 #   p2 = images[60:70]
 #   dataset = np.concatenate([images[0:60], images[70:140], images[150:]])
 #   
 #   test_significant_features(dataset, p1, p2, 
 #                             [5, 10, 15, 20, 60, 92], 150, 
 #                             use_PCA = True,
 #                             legend1 = 'p1_b', 
 #                             legend2 = 'p2_w') 
    
 #   random_person = images_unbearded[30:40]
 #   dataset = np.concatenate([images_unbearded[0:30], images_unbearded[40:]])
 #
 #   test_significant_features(dataset, random_person, images_bearded[10:20], 
 #                             [5, 10, 15, 20, 60, 92], 150, 
 #                             use_PCA = False,
 #                             legend1 = 'unbearded', 
 #                             legend2 = 'bearded') 

if __name__ == '__main__': 
    experimento_3c()
