import parser
from data_paths import faces_path
from figure_generation import save_eigenvector_figure
from pathlib import Path
from PCA import *
from PCA2D import PCA2D
from utilities import read_images

p = parser.create_parser("eigenvector_figure")
p.add_argument("--eigenface_grid_height",
               help="Eigenface grid height",
               type=int, default=2)
p.add_argument("--eigenface_grid_width",
               help="Eigenface grid width",
               type=int, default=2)
args = p.parse_args()
number_of_eigenvectors = args.number_of_eigenvectors
iterations = args.iterations
tolerance = args.tolerance

print("Reading images...")
images = read_images(Path(faces_path), args.scale_down_factor)

h, w = args.eigenface_grid_height, args.eigenface_grid_width
pca = PCA(h*w, iterations, tolerance)
pca2d = PCA2D(h*w, iterations, tolerance)
pca.fit(images)
pca2d.fit(images)
save_eigenvector_figure(pca, h, w, (8, 8))
save_eigenvector_figure(pca2d, h, w, (8, 8))