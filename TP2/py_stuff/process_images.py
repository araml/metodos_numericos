import matplotlib.pyplot as plt
from figure_generation import *
from data_paths import *
from pathlib import Path
from utilities import * 
from PCA import PCA
from PCA2D import PCA2D
import parser

p = parser.create_parser("process_images")
args = p.parse_args()
number_of_eigenvectors = args.number_of_eigenvectors
iterations = args.iterations
tolerance = args.tolerance
scale_down_factor = args.scale_down_factor
image_index = args.image_index
show_image_comparison = args.show_image_comparison

print("Reading images...")
images = read_images(Path(faces_path), scale_down_factor)

pca = PCA2D(number_of_eigenvectors, iterations, tolerance)
pca.fit(images)
create_pca_image_comparison(pca, images, image_index, colourmap=plt.cm.gray,
                            show_image=show_image_comparison)
