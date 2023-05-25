import matplotlib.pyplot as plt
import numpy as np
from figure_generation import *
from data_paths import *
from pathlib import Path
from utilities import * 
from PCA import PCA
from PCA2D import PCA2D
import parser

p = parser.create_parser()
args = p.parse_args()
number_of_eigenvectors = args.number_of_eigenvectors
iterations = args.iterations
tolerance = args.tolerance
scale_down_factor = args.scale_down_factor
print("number of eigenvectors: {}, iterations: {}, tolerance: {}".format(number_of_eigenvectors, iterations, tolerance))

print("Reading images...")
images = read_images(Path(faces_path), args.use_smaller_images, scale_down_factor)

pca = PCA2D(10, iterations, tolerance, filename="amogus")
pca.fit(images)
create_pca_image_comparison(pca, images, 10, (12, 12), plt.cm.magma)
#
#covariance_matrix = pca.create_covariance_matrix(pca.flatten_images(images))
#eigenvalues, _ = get_eigenvalues_and_eigenvectors(covariance_matrix, number_of_eigenvectors, iterations, tolerance)
#plot_eigenvalues(eigenvalues, "{}_eigenvalues".format(eigenvalues.size))
#
#

# pca_2d = PCA2D(15, iterations, tolerance, filename = "amogus")
# pca_2d.fit(images)
# compressed_image = pca_2d.transform(np.array(images[20]))
# plt.imshow(compressed_image)
# plt.show()


