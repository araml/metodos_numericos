import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from image_paths import faces_path 

print(faces_path)

paths = []
imgs = []

for path in sorted(list(Path(faces_path).rglob('*/*.pgm'))):
    paths.append(path)
    imgs.append(plt.imread(path))

square_images = np.stack(imgs[:4])
flattened_images = square_images.reshape(square_images.shape[0], square_images[0].size)
flattened_images = flattened_images - np.mean(flattened_images, axis=0) # subtract average from each
covariance = flattened_images.T @ flattened_images
n = covariance.shape[0]
covariance = covariance / (n-1)

la = np.linalg
e, v = la.eigh(covariance) # TODO: use our own implementation
