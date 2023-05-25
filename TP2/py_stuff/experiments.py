import numpy as np
from support import read_images
from pathlib import Path
from data_paths import * # TODO: rename data_paths since it also contains matrix path now
import matplotlib.pyplot as plt
import PCA from PCA 
import PCA2D from PCA2D

# No estoy entendiendo la dif entre este y 3b)
def ejercicio_3a() -> None:
    pass

# Ejercicio 3b)
# Maybe I'm not understanding this one, but should we reconstruct everyones 
# images and then compare them?
def similarity_analysis(one_person: np.array, rest: np.array, Ks: list) -> None
    for k in Ks:
        # Reconstruct with k 
        # single person/the rest 
        similarity_matrix = np.corrcoef(one_person, rest)
        plt.pcolor(similarity, cmap='GnBu')
        plt.title(f'Similarity matrix with k = {k}')
        plt.savefig(f'Similarity matrix with k = {k}')


# Ejercicio 3 c)
# No se bien como escribir Peak signal to noise ratio en imágenes RGB..
def quality_analysis(one_person: np.array, rest: np.array, bool pca_or_2dpca) -> None:
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
similarity = np.corrcoef(images[0:3].reshape(3, -1))
plt.pcolor(similarity, cmap='GnBu')
plt.show()

