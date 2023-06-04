from corrcoef_figure import run_corrcoef_experiment
from similarity_figure import run_similarity_experiment
from compression_error import run_compression_error_experiment
from plot_eigenvalues import plot_eigenvalues_experiment
from eigenvector_figure import run_eigenvector_figure_experiment

if __name__ == '__main__':
    run_corrcoef_experiment()
    run_similarity_experiment()
    run_error_compression_experiment()
    run_plot_eigenvalues_experiment()
    run_eigenvector_figure_experiment()
