import numpy as np
import matplotlib.pyplot as plt
import parser
from data_paths import * # TODO: rename data_paths since it also contains matrix path now
from figure_generation import *
from pathlib import Path
from PCA import *
from PCA2D import PCA2D
from utilities import read_images


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
# No se bien como escribir Peak signal to noise ratio en imágenes RGB..
def quality_analysis(one_person: np.array, rest: np.array, pca_engine) -> None:
    # TODO (Podríamos usar el mismo "entrenamiento" que para 3b tal vez)
    pca = None 
    if pca_or_2dpca:
        pca = PCA()
    else:
        pca = PCA2D(number_of_eigenvectors, iterations, tolerance, filename="amogus")

    pca.fit(rest)
    for original_image in one_person:
        compressed_image = pca.transform(np.array(original_image))
        # compare and graph
    


# example on how to use corrcoef
images = read_images(Path(faces_path), True, 8)
ejercicio_3a(PCA2D, images, 1, 2)
# similarity = np.corrcoef(images[0:100].reshape(100, -1))
# plt.pcolor(similarity, cmap='GnBu')
# plt.show()

# if __name__ == '__main__': 
#     parser = create_parser()
#     parser.parse_args()

#     images = read_images(Path(faces_path), args.usesmaller_images, 
#                          args.scale_down_factor)

#     quality_analysis(np.array(), images, True)
#     quality_analysis(np.array(), images, False)

#     similarity_analysis(np.array(), np.array(), [5, 20, 50])
