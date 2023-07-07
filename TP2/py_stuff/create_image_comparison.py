import matplotlib.pyplot as plt
import parser
from figure_generation import *
from data_paths import *
from pathlib import Path
from utilities import * 
from PCA import PCA
from PCA2D import PCA2D

p = parser.create_parser("create_image_comparison")
args = p.parse_args()
number_of_eigenvectors = args.number_of_eigenvectors
iterations = args.iterations
tolerance = args.tolerance
scale_down_factor = args.scale_down_factor
image_index = args.image_index
show_image_comparison = args.show_image_comparison
idxs = [6, 18, 27, 36, 42, 69, 120, 144, 180, 360]

print("Reading images...")
images = read_images(Path(faces_path), scale_down_factor)
max_2d_k = min(number_of_eigenvectors, images.shape[2])

ks_2d = [5, 10, 15, 20, 60, 92]
ks_1d = [20, 50, 100, 150, 250, 410]

create_compression_grid(PCA2D, images, idxs, ks_2d, iterations,
                        tolerance, 2, 3, (5,5), plt.cm.gray)
create_compression_grid(PCA, images, idxs, ks_1d, iterations,
                        tolerance, 2, 3, (5,5), plt.cm.gray)