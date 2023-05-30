import parser
from data_paths import *
from figure_generation import *
from pathlib import Path
from PCA import *
from PCA2D import PCA2D
from utilities import read_images

p = parser.create_parser("create_figure")
p.add_argument("--figure_function",
               help="Figure creation function to call",
               type=str)
args = p.parse_args()
number_of_eigenvectors = args.number_of_eigenvectors
iterations = args.iterations
tolerance = args.tolerance
scale_down_factor = args.scale_down_factor
image_index = args.image_index
show_image_comparison = args.show_image_comparison

print("Reading images...")
images = read_images(Path(faces_path), scale_down_factor)
max_2d_k = min(number_of_eigenvectors, images.shape[2])

figure_function = args.figure_function

if figure_function == "plot_eigenvalues":
    pca = PCA(number_of_eigenvectors, iterations, tolerance)
    pca2d = PCA2D(max_2d_k, iterations, tolerance)
    pca.fit(images)
    pca2d.fit(images)
    plot_eigenvalues(pca)
    plot_eigenvalues(pca2d)
elif figure_function == "save_eigenvector_figure":
    h, w = args.eigenface_grid_height, args.eigenface_grid_width
    pca = PCA(h*w, iterations, tolerance)
    pca2d = PCA2D(h*w, iterations, tolerance)
    pca.fit(images)
    pca2d.fit(images)
    save_eigenvector_figure(pca, h, w, (8, 8))
    save_eigenvector_figure(pca2d, h, w, (8, 8))
else:
    raise ValueError("Unknown figure function")