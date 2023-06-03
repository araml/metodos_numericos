import numpy as np
import parser
from data_paths import faces_path
from experiments import create_pca_similarity_figure
from pathlib import Path
from utilities import read_images

p = parser.create_parser("plot_eigenvalues")
args = p.parse_args()
number_of_eigenvectors = args.number_of_eigenvectors
iterations = args.iterations
tolerance = args.tolerance

print("Reading images...")
images = read_images(Path(faces_path), args.scale_down_factor)
max_2d_k = min(number_of_eigenvectors, images.shape[2])

k_range_2d = np.linspace(1, max_2d_k, 10, dtype=int)
k_range_2d_small = np.arange(1, 12)
k_range = np.linspace(1, 150, 10, dtype=int)
k_range_small = np.arange(1, 24)

for its in [1, 2, 3, 4, 5, 8, 10, 15]:
    create_pca_similarity_figure(images, k_range_2d, use_2d=True, iterations=its, plot_same=True)
    create_pca_similarity_figure(images, k_range_2d_small, use_2d=True, iterations=its, plot_same=False)
    create_pca_similarity_figure(images, k_range, use_2d=False, iterations=its, plot_same=True)
    create_pca_similarity_figure(images, k_range, use_2d=False, iterations=its, plot_same=False)
    create_pca_similarity_figure(images, k_range_small, use_2d=False, iterations=its, plot_same=False)