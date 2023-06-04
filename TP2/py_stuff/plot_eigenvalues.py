import parser
from data_paths import faces_path
from figure_generation import plot_eigenvalues
from pathlib import Path
from utilities import read_images

def plot_eigenvalues_experiment() -> None:
    p = parser.create_parser("plot_eigenvalues")
    args = p.parse_args()
    number_of_eigenvectors = args.number_of_eigenvectors
    iterations = args.iterations
    tolerance = args.tolerance

    print("Reading images...")
    images = read_images(Path(faces_path), args.scale_down_factor)

    plot_eigenvalues(images, number_of_eigenvectors, iterations, tolerance)

if __name__ == '__main__': 
    run_plot_eigenvalues_experiment()
