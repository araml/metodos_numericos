import numpy as np
from utilities import centre_images, flatten_images, get_eigenvalues_and_eigenvectors

class PCABase:
    def __init__(self, 
                 k: int = 30, 
                 iterations: int = 10, 
                 tolerance: float = 1e-17):
        self.eigenvectors = []
        self.eigenvalues = []
        self.k = k
        self.iterations = iterations
        self.tolerance = tolerance
        self.covariance_matrix = None

    def fit(self, images: np.array) -> None:
        raise NotImplementedError("Must be implemented in derived class")

    def transform(self, images: np.array) -> np.array:
        raise NotImplementedError("Must be implemented in derived class")
    
    def project_images(self, images: np.array) -> np.array:
        raise NotImplementedError("Must be implemented in derived class")
        
    def get_eigenfaces(self) -> np.array:
        raise NotImplementedError("Must be implemented in derived class")
    
    def get_image_comparison(self, images: np.array, 
                             image_index: int) -> (np.array, np.array):
        raise NotImplementedError("Must be implemented in derived class")
    
    def set_components_dimension(self, dimension: int) -> None:
        if self.k > len(self.eigenvalues):
            raise ValueError(f'Changed PCA components to {self.k} but max is'
                    '{len(self.eigenvalues)}, please rerun `fit` if you want to '
                    'use more components')
            
        self.k = dimension


class PCA(PCABase):
    def __init__(self, 
                 k: int = 30, 
                 iterations: int = 10, 
                 tolerance: float = 1e-17):
        super(PCA, self).__init__(k, iterations, tolerance)
        self.name = "1DPCA"

    def fit(self, images: np.array) -> None: 
        self.height, self.width = images[0].shape
        covariance = self.create_covariance_matrix(images)
        self.eigenvalues, self.eigenvectors = \
                get_eigenvalues_and_eigenvectors(covariance, 
                                                 self.k, 
                                                 self.iterations, 
                                                 self.tolerance)

    def transform(self, images: np.array) -> np.array:
        flattened_images = flatten_images(images)
        mean = np.mean(flattened_images, axis=0)
        projected_images = self.project_images(flattened_images-mean)
        return (projected_images @ self.eigenvectors[:, :self.k].T) + mean

    def create_covariance_matrix(self, images: np.array) -> np.array:
        if self.covariance_matrix is None:
            flattened_images = flatten_images(images)
            n = flattened_images.shape[0]
            centred_images = centre_images(flattened_images)
            self.covariance_matrix = (centred_images.T @ centred_images) / (n - 1)
        return self.covariance_matrix
    
    def get_eigenfaces(self) -> np.array:
        transposed_eigenvectors = self.eigenvectors.transpose()
        return np.array([e.reshape(self.height, self.width) for e in transposed_eigenvectors])
    
    def get_image_comparison(self, images: np.array, 
                             image_index: int) -> (np.array, np.array):
        compressed_images = self.transform(images)
        return images[image_index], compressed_images[image_index]
    
    def project_images(self, images: np.array) -> np.array:
        flattened_images = flatten_images(images)
        return flattened_images @ (self.eigenvectors[:, :self.k]) # reduce dimensions