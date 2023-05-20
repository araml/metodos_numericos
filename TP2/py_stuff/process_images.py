import argparse
import deflate as d
import matplotlib.pyplot as plt
import numpy as np
from image_paths import * # TODO: rename image_paths since it also contains matrix path now
from pathlib import Path
from random import choices
from string import ascii_letters
# from sklearn.decomposition import PCA
from utilities import * 
from PCA import PCA



# 2DPCA

def get_image_covariance_matrix(images: np.array) -> np.array:
    mean_pixel_values = np.mean(images, axis=0)
    centred_images = images - mean_pixel_values
    return np.mean(np.array([image.T @ image for image in centred_images]), axis=0)


def get_eigenbase_for_images(images: np.array,
                             k: int,
                             iters=10,
                             tolerance=1e-17,
                             filename=None,
                             get_all=False) -> (np.array, np.array):
    G = get_image_covariance_matrix(images)
    return get_eigenvalues_and_eigenvectors(G, k, iters, tolerance, filename, get_all)


def get_feature_vectors(image: np.array, eigenbase: np.array, k: int) -> np.array:
    return image @ eigenbase[:, :k]


def reconstruct_image_2DPCA(feature_vectors: np.array, eigenbase: np.array, k: int) -> np.array:
    return feature_vectors @ eigenbase[:, :k].T


def compress_single_image_2DPCA(image: np.array, eigenbase: np.array, k: int) -> np.array:
    feature_vectors = get_feature_vectors(image, eigenbase, k)
    return reconstruct_image_2DPCA(feature_vectors, eigenbase, k)


parser = argparse.ArgumentParser("process_images")
parser.add_argument("--use_smaller_images",
                    help="Decrease image resolution for faster computation time",
                    action="store_true")
parser.add_argument("--scale_down_factor",
                    help="Factor by which to scale down image resolution",
                    type=int, default=2)
parser.add_argument("--iterations",
                    help="Iterations for power method",
                    type=int, default=10)
parser.add_argument("--tolerance",
                    help="Tolerance for power method convergence criterion",
                    type=float, default=1e-17)

args = parser.parse_args()

print("Reading images...")
images = read_images(Path(faces_path), args.use_smaller_images, args.scale_down_factor)
h, w = images.shape[1], images.shape[2]
EIGENBASE_SIZE = 100

#print("Calculating eigenbase for 2DPCA...")
#eigenvalues, eigenbase = get_eigenbase_for_images(images,
#                                                  0,
#                                                  iters=args.iterations,
#                                                  tolerance=args.tolerance,
#                                                  get_all=True,
#                                                  filename="amogus")
#
#print("Compressing image...")
#compressed_image = compress_single_image_2DPCA(images[20], eigenbase, 20)
#plt.imshow(compressed_image)
#plt.show()



# f, axs = plt.subplots(3, 3, figsize=(12,12))
# for i, ax in enumerate(axs.flatten()):
#     ax.imshow(V[:,-i-1].reshape(h, w))
#     ax.axis('off')
# plt.tight_layout()
# plt.show()
# control_pca = PCA(n_components=h2*w2)
# pca_reduced = control_pca.fit_transform(flattened_images)
# pca_reconstructed = control_pca.inverse_transform(pca_reduced)

# plt.imshow(pca_reconstructed[1,:].reshape(h,w))
# plt.show()

pca = PCA()
pca.fit(images)
compressed_images = pca.transform(images)
print(compressed_images[0].shape)
plt.imshow(compressed_images[0].reshape(h, w))
plt.show()
