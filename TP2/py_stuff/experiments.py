import numpy as np
import matplotlib.pyplot as plt
from data_paths import * 
from figure_generation import *
from pathlib import Path
from PCA import *
from PCA2D import PCA2D
from utilities import read_images
from parser import create_parser

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
                     args,
                     person_outside_dataset: np.array = None,
                     pca_or_2d_pca = True) -> None:
    pca = None
    print(training_dataset[0].shape)
    h, w = training_dataset[0].shape[0], training_dataset[0].shape[1]
    if pca_or_2d_pca:
        pca = PCA(100, 100, args.tolerance)
    else:
        pca = PCA2D(args.number_of_eigenvectors, args.iterations, args.tolerance)

    pca.fit(training_dataset)
    results = []
    for original_image in person_inside_dataset:
        compressed_image = pca.transform(np.array([original_image]))
        # PCA flattens..
        if pca_or_2d_pca: 
            compressed_image = compressed_image.reshape(h, w)
        # frobenius norm by default
        print('Difference', np.linalg.norm(original_image - compressed_image))
        results.append(np.linalg.norm(original_image - compressed_image))
        
    print(results) 
    print(range(len(results)))
    plt.plot(range(len(results)), results)
    plt.show()



# example on how to use corrcoef
# similarity = np.corrcoef(images[0:100].reshape(100, -1))
# plt.pcolor(similarity, cmap='GnBu')
# plt.show()

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
    
    single_face = images[0:9]
    print(type(single_face))

#    excluded_face = read_images(Path(figures_experiments_path + '/cara_excluida'),
#            args.use_smaller_images, args.scale_down_factor)
   
    quality_analysis(images, single_face, args)
    #quality_analysis(np.array(), images, False)


