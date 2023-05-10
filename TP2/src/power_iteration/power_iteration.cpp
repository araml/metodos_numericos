#include <iostream>
#include <stdexcept>
#include <power_iteration.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;

// TODO(interops with python, can we use doubles?)
// TODO(return number of iterations until we converged)
//
std::tuple<float, VectorXd>
power_iteration_method(const MatrixXd &M, const VectorXd &x_0, size_t iters,
                float eps) {
    VectorXd v = x_0;
    VectorXd previous_v(v.size());
    previous_v.setOnes(v.size());
    previous_v += v;

    size_t i = 0;
    while (i < iters && (v - previous_v).norm() >= eps) {
        previous_v = v;
        auto Mv = M * v;
        v = Mv / Mv.norm();
        i++;
    }
    float lambda = v.transpose() * M * v;

    return std::make_tuple(lambda, v);
}

std::tuple<std::vector<float>, std::vector<VectorXd>>
deflate(MatrixXd M, const VectorXd &x_0, size_t iters, 
        size_t number_of_eigenvalues, float eps) { 
    
    if (M.cols() != M.rows() || M.cols() < number_of_eigenvalues) { 
        std::cout << "Matrix is " << M.cols() << "x" << M.rows() << " and asked for ";
        std::cout << number_of_eigenvalues << " eigenvalues" << std::endl;
        throw std::logic_error("Non square matrix or asked for too many eigenvalues");
    }

    std::vector<float> eigenvalues;
    std::vector<VectorXd> eigenvectors;
    for (size_t i = 0; i < number_of_eigenvalues; i++) {
        auto [l, v] = power_iteration_method(M, x_0, iters, eps);
        M = M - (l * v * v.transpose());
        eigenvalues.emplace_back(std::move(l));
        eigenvectors.emplace_back(std::move(v));
    }

    return std::make_tuple(eigenvalues, eigenvectors);
 }
