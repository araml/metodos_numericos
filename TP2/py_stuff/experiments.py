import numpy as np
import matplotlib.pyplot as plt
from data_paths import * 
from figure_generation import *
from pathlib import Path
from PCA import *
from PCA2D import PCA2D
from utilities import read_images
from parser import create_parser
import json

# No estoy entendiendo la dif entre este y 3b)
def ejercicio_3a(pca_class,
                 images: np.array,
                 small_k: int,
                 large_k: int,
                 iterations: int = 10,
                 tolerance: float = 1e-17,
                 filename: str = "amogus",
                 colourmap = plt.cm.GnBu) -> None:
    assert(large_k > small_k and small_k > 0)

    pca_engine = pca_class(large_k, iterations, tolerance, filename)
    pca_engine.fit(images)

    create_corrcoef_figure(pca_engine, images, colourmap)

    pca_engine.set_components_dimension(small_k)
    create_corrcoef_figure(pca_engine, images, colourmap)


# Ejercicio 3b)
# Maybe I'm not understanding this one, but should we reconstruct everyones 
# images and then compare them?
def similarity_analysis(one_person: np.array, rest: np.array, Ks: list) -> None:
    for k in Ks:
        # Reconstruct with k 
        # single person/the rest 
        similarity_matrix = np.corrcoef(one_person, rest)
        plt.pcolor(similarity, cmap='GnBu')
        plt.title(f'Similarity matrix with k = {k}')
        plt.savefig(f'Similarity matrix with k = {k}')


# Ejercicio 3 c)
def quality_analysis(training_dataset: np.array,
                     person_inside_dataset: np.array,
                     person_outside_dataset: np.array,
                     args,
                     pca_or_2d_pca = True) -> None:
    pca = None
    print(training_dataset[0].shape)
    h, w = training_dataset[0].shape[0], training_dataset[0].shape[1]
    if pca_or_2d_pca:
        pca = PCA(args.number_of_eigenvectors, args.iterations, args.tolerance)
    else:
        pca = PCA2D(args.number_of_eigenvectors, args.iterations, args.tolerance)

    pca.fit(training_dataset)
    results_in_dataset = []
    results_outside_dataset = []
    # im1 is inside the dataset, im2 is excluded
    for im1, im2 in zip(person_inside_dataset, person_outside_dataset):
        im1_compressed = pca.transform(np.array([im1]))
        im2_compressed = pca.transform(np.array([im2]))
        # PCA flattens..
        if pca_or_2d_pca: 
            im1_compressed = im1_compressed.reshape(h, w)
            im2_compressed = im2_compressed.reshape(h, w)
        # frobenius norm by default
        results_in_dataset.append(np.linalg.norm(im1 - im1_compressed))
        results_outside_dataset.append(np.linalg.norm(im2 - im2_compressed))
        
    print(results_in_dataset) 
    print(results_outside_dataset)
    plt.plot(range(len(results_in_dataset)), results_in_dataset, 'b')
    plt.plot(range(len(results_outside_dataset)), results_outside_dataset, 'r')
    plt.show()

    
# example on how to use corrcoef
# similarity = np.corrcoef(images[0:100].reshape(100, -1))
# plt.pcolor(similarity, cmap='GnBu')
# plt.show()

def plot_3c() -> None:
    results_in_dataset = [2132.9481474764225, 2136.4530386426954, 2182.0322211103885, 2157.0719449322187, 2137.33419595085, 2284.684520992702, 2247.1406287678465, 2185.737747347281, 2269.8563188969233]
    results_outside_dataset = [3130.754605656874, 3018.222367863016, 2794.223061783219, 2880.040997885756, 3179.5160795734723, 3100.2603989432564, 3037.8798743827842, 3139.3008201829543, 2855.938714383691]
    plt.plot(range(len(results_in_dataset)), results_in_dataset, 'b', label =
    'in_dataset')
    plt.plot(range(len(results_outside_dataset)), results_outside_dataset, 'r',
            label = 'out_dataset')
    plt.title('Comparación del error de compresion de PCA entre imagenes en el dataset y fuera del mismo')
    plt.legend()
    plt.show()
    plt.savefig('Comparacion PCA')

if __name__ == '__main__': 
    parser = create_parser()
    args = parser.parse_args()

    # Run excercise 3a
    #images = read_images(Path(faces_path), True, 8)
    #ejercicio_3a(PCA2D, images, 1, 2)

    # Run excercise 3b
    #similarity_analysis(np.array(), np.array(), [5, 20, 50])
    
    images = read_images(Path(faces_path), 
                         args.use_smaller_images, 
                         args.scale_down_factor)

    excluded_person = images[0:9]
    images = images[10:]
    single_person = images[0:9]
    
    plot_3c()
    #quality_analysis(images, single_person, excluded_person, args)
    #quality_analysis(np.array(), images, False)


