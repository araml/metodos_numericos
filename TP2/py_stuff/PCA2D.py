import numpy as np
from PCA import PCABase
from utilities import get_eigenvalues_and_eigenvectors 

class PCA2D(PCABase):
    def __init__(self, 
                 k: int, 
                 iterations: int = 10, 
                 tolerance: float = 1e-17,
                 filename: str = 'amogus'):
        super(PCA2D, self).__init__(k, iterations, tolerance, filename)
        self.name = "2DPCA"

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
    
    def get_image_comparison(self, images: np.array, 
                             image_index: int) -> (np.array, np.array):
        compressed_images = self.transform(images)
        return images[image_index], compressed_images[image_index]

    def compress_image(self, image: np.array) -> np.array:
        feature_vectors = image @ self.eigenvectors[:, :self.k]
        return feature_vectors @ self.eigenvectors[:, :self.k].T

    def get_image_covariance_matrix(self, images: np.array) -> np.array:
        mean_pixel_values = np.mean(images, axis = 0)
        centred_images = images - mean_pixel_values
        return np.mean(np.array([image.T @ image for image in centred_images]),
                       axis = 0)

    def set_components_dimension(self, dimension: int) -> None:
        if self.k > len(self.eigenvalues):
            raise ValueError(f'Changed 2DPCA components to {self.k} but max is'
                    '{len(self.eigenvalues)}, please rerun `fit` if you want to'
                    'use more components')
            
        self.k = dimension
