import numpy as np
import matplotlib.pyplot as plt

def read_images(path_to_images, use_smaller_images, scale_down_factor=1) -> np.array:
    paths = []
    images = []

    for path in sorted(list(path_to_images.rglob('*/*.pgm'))):
        paths.append(path)
        image = (plt.imread(path))
        if use_smaller_images:
            image = image[::scale_down_factor,::scale_down_factor] / 255
        images.append(image)

    return np.array(images)


