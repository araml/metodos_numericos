import argparse

def create_parser():
    parser = argparse.ArgumentParser("process_images")
    parser.add_argument("--use_smaller_images",
                        help="Decrease image resolution for faster computation time",
                        action="store_true")
    parser.add_argument("--scale_down_factor",
                        help="Factor by which to scale down image resolution",
                        type=int, default=2)
    parser.add_argument("--number_of_eigenvectors",
                        help="Number of eigenvectors to compute",
                        type=int, default=100)
    parser.add_argument("--number_of_components",
                        help="Number of components to use",
                        type=int, default=100)
    parser.add_argument("--iterations",
                        help="Iterations for power method",
                        type=int, default=10)
    parser.add_argument("--tolerance",
                        help="Tolerance for power method convergence criterion",
                        type=float, default=1e-17)

    return parser
