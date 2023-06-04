import parser
from data_paths import faces_path
from figure_generation import create_compression_corrcoef_figures, create_corrcoef_baseline
from pathlib import Path
from PCA import *
from PCA2D import PCA2D
from utilities import read_images

def run_corrcoef_experiment() -> None:
    p = parser.create_parser("corrcoef_figure")
    args = p.parse_args()
    number_of_eigenvectors = args.number_of_eigenvectors
    iterations = args.iterations
    tolerance = args.tolerance

    print("Reading images...")
    images = read_images(Path(faces_path), args.scale_down_factor)

    create_corrcoef_baseline(images)
    create_compression_corrcoef_figures(PCA2D, images, [1, 5, 40, 92], iterations, tolerance)
    create_compression_corrcoef_figures(PCA, images, [1, 10, 150, 400], iterations, tolerance)

if __name__ == '__main__': 
    run_corrcoef_experiment()
