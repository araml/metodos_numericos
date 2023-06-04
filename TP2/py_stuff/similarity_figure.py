import numpy as np
import parser
from pathlib import Path
from utilities import read_images
from data_paths import * 
from figure_generation import *
from pathlib import Path
from PCA import *
from PCA2D import PCA2D
from utilities import read_images
from parser import create_parser
from utilities import average_execution_time, centre_images
from threading import Thread
from multiprocessing import Process

SAMPLES_PER_PERSON = 10
PLOT_COLOURS = plt.rcParams['axes.prop_cycle'].by_key()['color']

def mean_similarity_between_people(correlation_matrix: np.array,
                                   person1: int,
                                   person2: int) -> float:
    start_row = person1 * SAMPLES_PER_PERSON
    end_row = start_row + SAMPLES_PER_PERSON
    start_column = person2 * SAMPLES_PER_PERSON
    end_column = start_column + SAMPLES_PER_PERSON
    submatrix = correlation_matrix[start_row:end_row, start_column:end_column]
    return submatrix.mean()


def compressed_mean_similarities(pca_engine: PCABase, images: np.array, k: int) -> (float, float):
    pca_engine.set_components_dimension(k)
    compressed_images = pca_engine.transform(images)
    return mean_similarities(compressed_images)


# Return mean of ALL similarities between photos of the same person
# and mean of ALL similarities between photos of two different people
def mean_similarities(images: np.array) -> (float, float):
    correlation_matrix = np.corrcoef(centre_images(images))
    number_of_people = correlation_matrix.shape[0] // SAMPLES_PER_PERSON
    mean_same_person_similarities = []
    mean_diff_person_similarities = []

    for i in range(number_of_people):
        mean_same_person_similarities.append(mean_similarity_between_people(correlation_matrix, i, i))
        current_diff_person_similarities = []
        for j in range(i+1, number_of_people):
            current_diff_person_similarities.append(mean_similarity_between_people(correlation_matrix, i, j))
        if len(current_diff_person_similarities):
            mean_diff_person_similarities.append(np.mean(current_diff_person_similarities))
    
    return (np.mean(mean_same_person_similarities), np.mean(mean_diff_person_similarities))


# For a given dataset, 2DPCA has an inherently lower max component limit,
# so we're making it optional -- this function can be run with just 1DPCA
# and a higher max_k
def create_pca_similarity_figure(images: np.array,
                                 ks: list,
                                 use_2d: bool,
                                 iterations: int = 10,
                                 tolerance: float = 1e-17,
                                 plot_same = True) -> None:
    max_k = max(ks)

    pca_1d = PCA(max_k, iterations, tolerance)
    pca_1d.fit(images)
    if use_2d:
        pca_2d = PCA2D(max_k, iterations, tolerance)
        pca_2d.fit(images)
    
    mean_similarities_1d, mean_similarities_2d = [], []
    baseline_same, baseline_diff = mean_similarities(images) # compare against uncompressed images

    for k in ks:
        mean_similarities_1d.append(compressed_mean_similarities(pca_1d, images, k))
        if use_2d:
            mean_similarities_2d.append(compressed_mean_similarities(pca_2d, images, k))

    _, axes = plt.subplots(figsize=(8, 6))

    if plot_same:
        axes.plot(ks, [x[0] for x in mean_similarities_1d], '-o', label='mismo, 1D')
    axes.plot(ks, [x[1] for x in mean_similarities_1d], '-o', label='distintos, 1D', color=PLOT_COLOURS[1])
    if use_2d:
        if plot_same:
            axes.plot(ks, [x[0] for x in mean_similarities_2d], '-o', label='mismo, 2D', color=PLOT_COLOURS[2])
        axes.plot(ks, [x[1] for x in mean_similarities_2d], '-o', label='distintos, 2D', color=PLOT_COLOURS[3])
    if plot_same:
        axes.plot(ks, [baseline_same] * len(ks), '--', label="mismo, sin comprimir", color=PLOT_COLOURS[4])
    axes.plot(ks, [baseline_diff] * len(ks), '--', label="distintos, sin comprimir", color=PLOT_COLOURS[5])

    plt.xlabel("Componentes usadas")
    plt.ylabel("Similaridad promedio")
    plt.title(f"Similaridad promedio entre imágenes de dimensión {images[0].shape}\ncon {iterations} iteraciones y tolerancia {tolerance}")
    plt.xticks(ks)
    if plot_same:
        plt.ylim(bottom = -0.1, top = 1.1)
    plt.legend()
    file_path = Path(figures_path, f"similaridad_{iterations}iteraciones_tolerancia{tolerance}_dim{images[0].shape}_2d{use_2d}_max{max_k}_mismo{plot_same}.png")
    plt.savefig(file_path)
    plt.clf()

def run_similarity_experiment() -> None:
    p = parser.create_parser("plot_eigenvalues")
    args = p.parse_args()
    number_of_eigenvectors = args.number_of_eigenvectors
    iterations = args.iterations
    tolerance = args.tolerance

    print("Reading images...")
    images = read_images(Path(faces_path), args.scale_down_factor)
    max_2d_k = min(number_of_eigenvectors, images.shape[2])

    k_range_2d = np.linspace(1, max_2d_k, 10, dtype=int)
    k_range_2d_small = np.arange(1, 12)
    k_range = np.linspace(1, 150, 10, dtype=int)
    k_range_small = np.arange(1, 24)

    for its in [1, 2, 3, 4, 5, 8, 10, 15]:
        create_pca_similarity_figure(images, k_range_2d, use_2d=True, iterations=its, plot_same=True)
        create_pca_similarity_figure(images, k_range_2d_small, use_2d=True, iterations=its, plot_same=False)
        create_pca_similarity_figure(images, k_range, use_2d=False, iterations=its, plot_same=True)
        create_pca_similarity_figure(images, k_range, use_2d=False, iterations=its, plot_same=False)
        create_pca_similarity_figure(images, k_range_small, use_2d=False, iterations=its, plot_same=False)

if __name__ == '__main__': 
    run_similarity_experiment()
