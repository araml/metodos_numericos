import argparse

def create_parser(prog: str):
    parser = argparse.ArgumentParser(prog)
    parser.add_argument("--scale_down_factor",
                        help="Factor by which to scale down image resolution",
                        type=int, default=1)
    parser.add_argument("--number_of_eigenvectors",
                        help="Number of eigenvectors to compute",
                        type=int, default=20)
    parser.add_argument("--iterations",
                        help="Iterations for power method",
                        type=int, default=10)
    parser.add_argument("--tolerance",
                        help="Tolerance for power method convergence criterion",
                        type=float, default=1e-17)
    parser.add_argument("--image_index",
                        help="Number of image to use for comparison",
                        type=int, default=69)
    parser.add_argument("--similarity_2dpca",
                        help="Whether or not to plot 2DPCA in similarity figure",
                        action="store_true")
    parser.add_argument("--show_image_comparison",
                        help="Whether or not to display image comparison before saving it",
                        action="store_true")

    return parser
