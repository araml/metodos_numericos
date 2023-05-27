import numpy as np
import matplotlib.pyplot as plt
from data_paths import * 
from figure_generation import *
from pathlib import Path
from PCA import *
from PCA2D import PCA2D
from utilities import read_images
from parser import create_parser
from utilities import flatten_images, average_execution_time

SAMPLES_PER_PERSON = 10

def ejercicio_3a(pca_class,
                 images: np.array,
                 small_k: int,
                 large_k: int,
                 iterations: int = 10,
                 tolerance: float = 1e-17,
                 colourmap = plt.cm.GnBu) -> None:
    assert(large_k > small_k and small_k > 0)

    pca_engine = pca_class(large_k, iterations, tolerance)
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


def compressed_mean_similarities(pca_engine: PCABase, images: np.array, k: int) -> (float, float):
    pca_engine.set_components_dimension(k)
    compressed_images = pca_engine.transform(images)
    return mean_similarities(compressed_images)


def mean_similarities(images: np.array) -> (float, float):
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
    baseline_same, baseline_diff = mean_similarities(images) # compare against uncompressed images

    for k in Ks:
        same_1d, diff_1d = compressed_mean_similarities(pca_1d, images, k)
        same_2d, diff_2d = compressed_mean_similarities(pca_2d, images, k)
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
    plt.title("Similaridad promedio entre imágenes de dimensión {}\ncon {} iteraciones y tolerancia {}".format(images[0].shape, iterations, tolerance))
    plt.xticks(Ks)
    plt.ylim(bottom=0.0)
    plt.legend()
    file_path = Path(figures_path, "similaridad_{}iteraciones_tolerancia{}_dim{}.png".format(iterations, tolerance, images[0].shape))
    plt.savefig(file_path)


# Ejercicio 3 c)
def PSNR(m1: np.array, m2: np.array) -> float:
    mse = (np.square(m1 - m2)).mean()
    return 20 * np.log10(255 / np.sqrt(mse))

def normalize(m1: np.array) -> np.array:
    return m1 / np.linalg.norm(m1)

def quality_analysis(training_dataset: np.array,
                     person_inside_dataset: np.array,
                     person_outside_dataset: np.array,
                     ks: list = [10], 
                     iterations: int = 100, 
                     use_PCA = True) -> None:
    pca = None
    print(training_dataset[0].shape)
    h, w = training_dataset[0].shape[0], training_dataset[0].shape[1]
    # get max number of eigenvalues for training
    k = max(ks)
    if use_PCA:
        pca = PCA(k, iterations)
    else:
        pca = PCA2D(k, iterations)

    pca.fit(training_dataset)
    frobenius_error_in_dataset = []
    psnr_error_in_dataset = []
    frobenius_error_outside_dataset = []
    psnr_error_outside_dataset = []
    # im1 is inside the dataset, im2 is excluded
    for k in ks: 
        r1, r2, p1, p2 = [], [], [], []
        pca.set_components_dimension(k)
        for im1, im2 in zip(person_inside_dataset, person_outside_dataset):
            im1_compressed = pca.transform(np.array([im1]))
            im2_compressed = pca.transform(np.array([im2]))
            # PCA flattens..
            if use_PCA: 
                im1_compressed = im1_compressed.reshape(h, w)
                im2_compressed = im2_compressed.reshape(h, w)
            # frobenius norm by default
            r1.append(np.linalg.norm(normalize(im1) - normalize(im1_compressed)))
            p1.append(PSNR(im1, im1_compressed))
            r2.append(np.linalg.norm(normalize(im2) - normalize(im2_compressed)))
            p2.append(PSNR(im2, im2_compressed))
        frobenius_error_in_dataset.append(r1)
        psnr_error_in_dataset.append([x / max(p1) for x in p1])
        frobenius_error_outside_dataset.append(r2)
        psnr_error_outside_dataset.append([x / max(p2) for x in p2])
        

    print(frobenius_error_in_dataset) 
    print(psnr_error_in_dataset)
    print(frobenius_error_outside_dataset)
    print(psnr_error_outside_dataset)
    
    
# example on how to use corrcoef
# similarity = np.corrcoef(images[0:100].reshape(100, -1))
# plt.pcolor(similarity, cmap='GnBu')
# plt.show()

def plot_3c(results_in_dataset, results_outside_dataset, pca_or_2d_pca = 'PCA') -> None:
    plt.plot(range(len(results_in_dataset)), results_in_dataset, 'b', label =
    'in_dataset')
    plt.plot(range(len(results_outside_dataset)), results_outside_dataset, 'r',
            label = 'out_dataset')
    plt.title('Comparación de error usando ' + pca_or_2d_pca)
    plt.legend()
 #   plt.show()
    plt.savefig('Comparacion ' + pca_or_2d_pca + '.png')

def ejercicio_3d(images, Ks, repetitions):
    times_1d = []
    times_2d = []
    for k in Ks:
        pca_1d = PCA(k, iterations=5)
        pca_2d = PCA2D(k, iterations=5)
        t_1d = average_execution_time(pca_1d.fit, repetitions, images)
        t_2d = average_execution_time(pca_2d.fit, repetitions, images)
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
    parser = create_parser("experiments")
    args = parser.parse_args()
    
    images = read_images(Path(faces_path), 
                         args.scale_down_factor)
    
    # Run exercise 3a
    #ejercicio_3a(PCA2D, images, 1, 2)

    #max_components = min(images[0].shape)
    #k_range = np.linspace(1, max_components, 8, dtype=int)

    # excluded_person = images[0:9]
    # images = images[10:]
    # single_person = images[0:9]

    excluded_person = images[0:9]
    images = images[10:]
    single_person = images[0:9]
    
    #quality_analysis(images, single_person, excluded_person)
    # Runs 2DPCA
    quality_analysis(images, single_person, excluded_person, [10], 10, False)


    # pca = PCA2D(40, filename="amogus")
    # pca.fit(images)
    #for its in [1, 2, 3, 4, 5, 8, 10, 15, 20]:
    #    ejercicio_3b(images, k_range, its)
    # # ejercicio_3d(images, k_range, 50)
    # print(images.shape)
    #quality_analysis(images, single_person, excluded_person, args)
    #quality_analysis(np.array(), images, False)
