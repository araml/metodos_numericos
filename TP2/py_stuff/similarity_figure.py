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

k_range_linear = np.linspace(1, number_of_eigenvectors, 10, dtype=int)
k_range_2d_linear = np.linspace(1, max_2d_k, 10, dtype=int)
k_range_log = np.arange(24)
k_range_2d_log = np.arange(12)
for its in [1, 2, 3, 4, 5, 8, 10, 15, 20]:
    create_pca_similarity_figure(images, k_range_2d_log, use_2d=True, iterations=its, scale="log")
    create_pca_similarity_figure(images, k_range_2d_linear, use_2d=True, iterations=its, scale="linear")
    create_pca_similarity_figure(images, k_range_log, use_2d=False, iterations=its, scale="log")
    create_pca_similarity_figure(images, k_range_linear, use_2d=False, iterations=its, scale="linear")