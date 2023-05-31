import parser
from data_paths import faces_path
from figure_generation import create_compression_corrcoef_figures
from pathlib import Path
from PCA import *
from PCA2D import PCA2D
from utilities import read_images

p = parser.create_parser("corrcoef_figure")
p.add_argument("--large_k",
               help="Larger k value for corrcoef figure generation",
               type=int, default=10)
p.add_argument("--small_k",
               help="Smaller k value for corrcoef figure generation",
               type=int, default=1)
args = p.parse_args()
number_of_eigenvectors = args.number_of_eigenvectors
iterations = args.iterations
tolerance = args.tolerance

print("Reading images...")
images = read_images(Path(faces_path), args.scale_down_factor)

large_k, small_k = args.large_k, args.small_k
create_compression_corrcoef_figures(PCA, images, small_k, large_k, iterations, tolerance)
if large_k >= images.shape[2]:
    create_compression_corrcoef_figures(PCA2D, images, small_k, large_k, iterations, tolerance)