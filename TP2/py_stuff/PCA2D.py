import numpy as np 
from utilities import get_eigenvalues_and_eigenvectors 

class PCA2D:
    def __init__(self, 
                 k: int, 
                 iterations: int = 10, 
                 tolerance: float = 1e-17,
                 filename: str = None):
        self.eigenvectors = []
        self.eigenvalues = []
        self.k = k
        self.tolerance = tolerance
        self.iterations = iterations
        self.filename = filename

    def fit(self, images: np.array) -> None:
            G = self.get_image_covariance_matrix(images)
            self.eigenvalues, self.eigenvectors = \
                get_eigenvalues_and_eigenvectors(G, self.k, self.iterations,
                                                 self.tolerance, self.filename)

    def transform(self, images: np.array) -> np.array:
        compressed_images = []
        for image in images: 
            compressed_images.append(self.compress_image(image))

        return compressed_images

    def compress_image(self, image: np.array) -> np.array:
        feature_vectors = image @ self.eigenvectors[:, :self.k]
        return feature_vectors @ self.eigenvectors[:, :self.k].T

    def get_image_covariance_matrix(self, images: np.array) -> np.array:
        mean_pixel_values = np.mean(images, axis = 0)
        centred_images = images - mean_pixel_values
        return np.mean(np.array([image.T @ image for image in centred_images]),
                       axis = 0)
