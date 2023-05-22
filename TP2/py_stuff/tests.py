import unittest
from PCA import PCA 
from PCA2D import PCA2D

class testPCAMethods(unittest.TestCase):
    def test_PCA_less_than_available_dimensions(self):
        pca = PCA()
        with self.assertRaises(ValueError):
            pca.set_components_dimension(60)

    def test_PCA2D_less_than_available_dimensions(self):
        pca = PCA2D(15)
        with self.assertRaises(ValueError):
            pca.set_components_dimension(60)

if __name__ == '__main__':
    unittest.main()
