import numpy as np
from support import read_images
from pathlib import Path
from data_paths import * # TODO: rename data_paths since it also contains matrix path now
import matplotlib.pyplot as plt

def similarity_matrix(data: np.array, similarity_function) -> np.array:
    return np.corrcoef(data)

images = read_images(Path(faces_path), True, 8)
similarity = similarity_matrix(images[0:3].reshape(3, -1), lambda x,y: x)
plt.pcolor(similarity, cmap='GnBu')
plt.show()

