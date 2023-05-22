import argparse
import matplotlib.pyplot as plt
import numpy as np
from figure_generation import *
from data_paths import *
from pathlib import Path
from utilities import * 
from PCA import PCA
from PCA2D import PCA2D


# 2DPCA

parser = argparse.ArgumentParser("process_images")
parser.add_argument("--use_smaller_images",
                    help="Decrease image resolution for faster computation time",
                    action="store_true")
parser.add_argument("--scale_down_factor",
                    help="Factor by which to scale down image resolution",
                    type=int, default=2)
parser.add_argument("--number_of_eigenvectors",
                    help="Number of eigenvectors to compute",
                    type=int, default=100)
parser.add_argument("--number_of_components",
                    help="Number of components to use",
                    type=int, default=100)
parser.add_argument("--iterations",
                    help="Iterations for power method",
                    type=int, default=10)
parser.add_argument("--tolerance",
                    help="Tolerance for power method convergence criterion",
                    type=float, default=1e-17)

args = parser.parse_args()
number_of_eigenvectors = args.number_of_eigenvectors
number_of_components = args.number_of_components
iterations = args.iterations
tolerance = args.tolerance
scale_down_factor = args.scale_down_factor
print("number of eigenvectors: {}, iterations: {}, tolerance: {}".format(number_of_eigenvectors, iterations, tolerance))

print("Reading images...")
images = read_images(Path(faces_path), args.use_smaller_images, scale_down_factor)

#pca = PCA(number_of_eigenvectors, iterations, tolerance)
#
#covariance_matrix = pca.create_covariance_matrix(pca.flatten_images(images))
#eigenvalues, _ = get_eigenvalues_and_eigenvectors(covariance_matrix, number_of_eigenvectors, iterations, tolerance)
#plot_eigenvalues(eigenvalues, "{}_eigenvalues".format(eigenvalues.size))
#
#

pca_2d = PCA2D(15, iterations, tolerance, filename = "amogus")
pca_2d.fit(images)
compressed_image = pca_2d.transform(np.array(images[20]))
plt.imshow(compressed_image)
plt.show()
