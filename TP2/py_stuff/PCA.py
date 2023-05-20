import numpy as np
from utilities import get_eigenvalues_and_eigenvectors

class PCA:
    def __init__(self, number_of_components: int = 30, iterations: int = 10, tolerance: float = 1e-17):
        self.eigenbase = []
        self.eigenvalues = []
        self.number_of_components = number_of_components
        self.iterations = iterations
        self.tolerance = tolerance

        self.flattened_images = None

    # TODO
    def change_PCA_dimension(self, dimension: int) -> None:
        pass

    def flatten_images(self, images: np.array) -> np.array:
        square_images = np.stack(images)
        flattened_images = square_images.reshape(square_images.shape[0], square_images[0].size)
        return flattened_images

    def fit(self, images: np.array, filename = None) -> None: 
        self.flattened_images = self.flatten_images(images)
        covariance = self.create_covariance_matrix(self.flattened_images)
        self.eigenvalues, self.eigenbase = get_eigenvalues_and_eigenvectors(covariance, self.number_of_components, self.iterations, self.tolerance, filename)

    def transform(self, images: np.array, k: int = 30) -> np.array:
        flattened_images = self.flatten_images(images)
        reduced_images = flattened_images @ (self.eigenbase[:, :k]) # reduce dimensions
        return reduced_images @ self.eigenbase[:, :k].T

    def create_covariance_matrix(self, flattened_images: np.array) -> np.array:
        n = flattened_images.shape[0]
        # subtract average from each
        centred_images = flattened_images - np.mean(flattened_images, axis=0) 
        return (centred_images.T @ centred_images) / (n - 1)
