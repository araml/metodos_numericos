import numpy as np
import matplotlib.pyplot as plt
from data_paths import * 
from figure_generation import *
from pathlib import Path
from PCA import *
from PCA2D import PCA2D
from utilities import read_images
from parser import create_parser
from utilities import flatten_images, get_average_execution_time

SAMPLES_PER_PERSON = 10

def ejercicio_3a(pca_class,
                 images: np.array,
                 small_k: int,
                 large_k: int,
                 iterations: int = 10,
                 tolerance: float = 1e-17,
                 filename: str = "amogus",
                 colourmap = plt.cm.GnBu) -> None:
    assert(large_k > small_k and small_k > 0)

    pca_engine = pca_class(large_k, iterations, tolerance, filename)
    pca_engine.fit(images)

    create_corrcoef_figure(pca_engine, images, colourmap)

    pca_engine.set_components_dimension(small_k)
    create_corrcoef_figure(pca_engine, images, colourmap)


def mean_similarity_between_people(correlation_matrix: np.array,
                                   person1: int,
                                   person2: int) -> float:
    start_row = person1 * SAMPLES_PER_PERSON
    end_row = start_row + SAMPLES_PER_PERSON
    start_column = person2 * SAMPLES_PER_PERSON
    end_column = start_column + SAMPLES_PER_PERSON
    submatrix = correlation_matrix[start_row:end_row, start_column:end_column]
    return submatrix.mean()


def get_compressed_mean_similarities(pca_engine: PCABase, images: np.array, k: int) -> (float, float):
    pca_engine.set_components_dimension(k)
    compressed_images = pca_engine.transform(images)
    return get_mean_similarities(compressed_images)


def get_mean_similarities(images: np.array) -> (float, float):
    correlation_matrix = np.corrcoef(flatten_images(images))
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


def ejercicio_3b(images: np.array, Ks: list, iterations: int = 10, tolerance: float = 1e-17) -> None:
    max_k = max(Ks)

    pca_1d = PCA(max_k, iterations, tolerance)
    pca_2d = PCA2D(max_k, iterations, tolerance)
    pca_1d.fit(images)
    pca_2d.fit(images)
    
    mean_same_person_similarities_1d = []
    mean_diff_person_similarities_1d = []
    mean_same_person_similarities_2d = []
    mean_diff_person_similarities_2d = []
    baseline_same, baseline_diff = get_mean_similarities(images) # compare against uncompressed images

    for k in Ks:
        same_1d, diff_1d = get_compressed_mean_similarities(pca_1d, images, k)
        same_2d, diff_2d = get_compressed_mean_similarities(pca_2d, images, k)
        mean_same_person_similarities_1d.append(same_1d)
        mean_diff_person_similarities_1d.append(diff_1d)
        mean_same_person_similarities_2d.append(same_2d)
        mean_diff_person_similarities_2d.append(diff_2d)

    _, axes = plt.subplots(figsize=(8, 6))

    axes.plot(Ks, mean_same_person_similarities_1d, '-o', label='mismo, 1D')
    axes.plot(Ks, mean_diff_person_similarities_1d, '-o', label='distintos, 1D')
    axes.plot(Ks, mean_same_person_similarities_2d, '-o', label='mismo, 2D')
    axes.plot(Ks, mean_diff_person_similarities_2d, '-o', label='distintos, 2D')
    axes.plot(Ks, [baseline_same] * len(Ks), '--', label="mismo, sin comprimir")
    axes.plot(Ks, [baseline_diff] * len(Ks), '--', label="distintos, sin comprimir")

    plt.xlabel("Componentes usadas")
    plt.ylabel("Similaridad promedio")
    plt.title("Similaridad promedio entre imágenes de una misma persona\ny de distintas personas para PCA y 2DPCA\ncon {} iteraciones y tolerancia {}".format(iterations, tolerance))
    plt.xticks(Ks)
    plt.ylim(bottom=0.0)
    plt.legend()
    file_path = Path(figures_path, "similaridad_{}iteraciones_tolerancia{}.png".format(iterations, tolerance))
    plt.savefig(file_path)


# Ejercicio 3 c)
def quality_analysis(training_dataset: np.array,
                     person_inside_dataset: np.array,
                     person_outside_dataset: np.array,
                     args,
                     use_1d = True) -> None:
    pca = None
    print(training_dataset[0].shape)
    h, w = training_dataset[0].shape[0], training_dataset[0].shape[1]
    if use_1d:
        pca = PCA(args.number_of_eigenvectors, args.iterations, args.tolerance)
    else:
        pca = PCA2D(args.number_of_eigenvectors, args.iterations, args.tolerance)

    pca.fit(training_dataset)
    results_in_dataset = []
    results_outside_dataset = []
    # im1 is inside the dataset, im2 is excluded
    for im1, im2 in zip(person_inside_dataset, person_outside_dataset):
        im1_compressed = pca.transform(np.array([im1]))
        im2_compressed = pca.transform(np.array([im2]))
        # PCA flattens..
        if use_1d: 
            im1_compressed = im1_compressed.reshape(h, w)
            im2_compressed = im2_compressed.reshape(h, w)
        # frobenius norm by default
        results_in_dataset.append(np.linalg.norm(im1 - im1_compressed))
        results_outside_dataset.append(np.linalg.norm(im2 - im2_compressed))
        
    print(results_in_dataset) 
    print(results_outside_dataset)
    plt.plot(range(len(results_in_dataset)), results_in_dataset, 'b')
    plt.plot(range(len(results_outside_dataset)), results_outside_dataset, 'r')
    plt.show()

    
# example on how to use corrcoef
# similarity = np.corrcoef(images[0:100].reshape(100, -1))
# plt.pcolor(similarity, cmap='GnBu')
# plt.show()

def plot_3c() -> None:
    results_in_dataset = [2132.9481474764225, 2136.4530386426954, 2182.0322211103885, 2157.0719449322187, 2137.33419595085, 2284.684520992702, 2247.1406287678465, 2185.737747347281, 2269.8563188969233]
    results_outside_dataset = [3130.754605656874, 3018.222367863016, 2794.223061783219, 2880.040997885756, 3179.5160795734723, 3100.2603989432564, 3037.8798743827842, 3139.3008201829543, 2855.938714383691]
    plt.plot(range(len(results_in_dataset)), results_in_dataset, 'b', label =
    'in_dataset')
    plt.plot(range(len(results_outside_dataset)), results_outside_dataset, 'r',
            label = 'out_dataset')
    plt.title('Comparación del error de compresion de PCA entre imagenes en el dataset y fuera del mismo')
    plt.legend()
    plt.show()
    plt.savefig('Comparacion PCA')


def ejercicio_3d(images, Ks, repetitions):
    times_1d = []
    times_2d = []
    for k in Ks:
        pca_1d = PCA(k)
        pca_2d = PCA2D(k)
        t_1d = get_average_execution_time(pca_1d.fit, repetitions, images)
        t_2d = get_average_execution_time(pca_2d.fit, repetitions, images)
        times_1d.append(t_1d)
        times_2d.append(t_2d)

    _, axes = plt.subplots(figsize=(8, 6))
    axes.plot(Ks, times_1d, '-o', label="PCA")
    axes.plot(Ks, times_2d, '-o', label="2DPCA")
    plt.xticks(Ks)
    plt.xlabel("Autovectores calculados")
    plt.ylabel("Tiempo de ejecución (en segundos)")
    plt.title("Tiempo de ejecución en función de la cantidad de autovectores calculados,\npromedio sobre {} repeticiones".format(repetitions))
    plt.legend()
    file_path = Path(figures_path, "tiempo_{}repeticiones_dim{}.png".format(repetitions, images[0].shape))
    plt.savefig(file_path)

if __name__ == '__main__': 
    parser = create_parser()
    args = parser.parse_args()

    # Run excercise 3a
    #images = read_images(Path(faces_path), True, 8)
    #ejercicio_3a(PCA2D, images, 1, 2)

    # Run excercise 3b
    #similarity_analysis(np.array(), np.array(), [5, 20, 50])
    
    images = read_images(Path(faces_path), 
                         args.use_smaller_images, 
                         args.scale_down_factor)
    
    max_components = min(images[0].shape)
    k_range = np.linspace(1, max_components, 10, dtype=int)

    # excluded_person = images[0:9]
    # images = images[10:]
    # single_person = images[0:9]
    
    # plot_3c()
    # pca = PCA2D(40, filename="amogus")
    # pca.fit(images)
    # for its in [1, 2, 3, 4, 5, 8, 10, 15, 20]:
    #     ejercicio_3b(images, k_range, its)
    ejercicio_3d(images, k_range, 50)
    # print(images.shape)
    #quality_analysis(images, single_person, excluded_person, args)
    #quality_analysis(np.array(), images, False)


