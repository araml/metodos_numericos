import numpy as np
from utilities import get_eigenvalues_and_eigenvectors

class PCABase:
    def __init__(self, 
                 k: int = 30, 
                 iterations: int = 10, 
                 tolerance: float = 1e-17,
                 filename: str = None):
        self.eigenbase = []
        self.eigenvalues = []
        self.k = k
        self.iterations = iterations
        self.tolerance = tolerance
        self.filename = filename

    def fit(self, images: np.array) -> None:
        raise NotImplementedError("Must be implemented in derived class")

    def transform(self, images: np.array) -> np.array:
        raise NotImplementedError("Must be implemented in derived class")
    
    def get_image_comparison(self, images: np.array, 
                             image_index: int) -> (np.array, np.array):
        raise NotImplementedError("Must be implemented in derived class")


class PCA(PCABase):
    def __init__(self, 
                 k: int = 30, 
                 iterations: int = 10, 
                 tolerance: float = 1e-17,
                 filename: str = None):
        super(PCA, self).__init__(k, iterations, tolerance, filename)
        self.name = "1DPCA"

    def fit(self, images: np.array) -> None: 
        flattened_images = self.flatten_images(images)
        covariance = self.create_covariance_matrix(flattened_images)
        self.eigenvalues, self.eigenbase = \
                get_eigenvalues_and_eigenvectors(covariance, 
                                                 self.k, 
                                                 self.iterations, 
                                                 self.tolerance, self.filename)

    def transform(self, images: np.array) -> np.array:
        flattened_images = self.flatten_images(images)
        reduced_images = flattened_images @ (self.eigenbase[:, :self.k]) # reduce dimensions
        return reduced_images @ self.eigenbase[:, :self.k].T

    def create_covariance_matrix(self, flattened_images: np.array) -> np.array:
        n = flattened_images.shape[0]
        # subtract average from each
        centred_images = flattened_images - np.mean(flattened_images, axis=0)
        return (centred_images.T @ centred_images) / (n - 1)
    
    def get_image_comparison(self, images: np.array, 
                             image_index: int) -> (np.array, np.array):
        compressed_images = self.transform(images)
        return images[image_index], compressed_images[image_index]

    def flatten_images(self, images: np.array) -> np.array:
        square_images = np.stack(images)
        flattened_images = square_images.reshape(square_images.shape[0], 
                                                 square_images[0].size)
        return flattened_images

    def set_components_dimension(self, dimension: int) -> None:
        if self.k > len(self.eigenvalues):
            raise ValueError(f'Changed PCA components to {self.k} but max is'
                    '{len(self.eigenvalues)}, please rerun `fit` if you want to'
                    'use more components')
            
        self.k = dimension
