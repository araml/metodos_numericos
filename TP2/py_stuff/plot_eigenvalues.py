import parser
from data_paths import faces_path
from figure_generation import plot_eigenvalues
from pathlib import Path
from PCA import *
from PCA2D import PCA2D
from utilities import read_images

p = parser.create_parser("plot_eigenvalues")
args = p.parse_args()
number_of_eigenvectors = args.number_of_eigenvectors
iterations = args.iterations
tolerance = args.tolerance

print("Reading images...")
images = read_images(Path(faces_path), args.scale_down_factor)

pca = PCA(number_of_eigenvectors, iterations, tolerance)
pca.fit(images)
plot_eigenvalues(pca)

if number_of_eigenvectors >= images.shape[2]:
    pca2d = PCA2D(number_of_eigenvectors, iterations, tolerance)
    pca2d.fit(images)
    plot_eigenvalues(pca2d)