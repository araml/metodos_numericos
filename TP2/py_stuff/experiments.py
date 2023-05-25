import numpy as np
from support import read_images
from pathlib import Path
from data_paths import * # TODO: rename data_paths since it also contains matrix path now
import matplotlib.pyplot as plt

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


# example on how to use corrcoef
images = read_images(Path(faces_path), True, 8)
similarity = np.corrcoef(images[0:3].reshape(3, -1))
plt.pcolor(similarity, cmap='GnBu')
plt.show()

