#include <iostream>
#include <stdexcept>
#include <power_iteration.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;

// TODO(interops with python, can we use doubles?)
// TODO(use eps to see if we converge faster)
// TODO(return number of iterations until we converged)
//
std::tuple<float, VectorXd>
metodo_potencia(const MatrixXd &M, const VectorXd &x_0, size_t iters,
                float eps) {
    VectorXd v = x_0;
    for (size_t i = 0; i < iters; i++) {
        auto bv = M * v;
        v = bv / bv.norm();
    }
    float lambda = v.transpose() * M * v;

    return std::make_tuple(lambda, v);
}

std::tuple<std::vector<float>, std::vector<VectorXd>>
deflate(MatrixXd M, const VectorXd &x_0, size_t iters, 
        size_t number_of_eigenvalues, float eps) { 
    
    if (M.cols() != M.rows() || M.cols() < number_of_eigenvalues) { 
        std::cout << "Matrix is " << M.cols() << "x" << M.rows() << " and asked for ";
        std::cout << number_of_eigenvalues << "eigenvalues" << std::endl;
        throw std::logic_error("Non squared matrix or asked for too many eigenvalues");
    }

    std::vector<float> eigenvalues;
    std::vector<VectorXd> eigenvectors;
    for (size_t i = 0; i < number_of_eigenvalues; i++) { 
        auto [l, v] = metodo_potencia(M, x_0, iters, eps);
        M = M - (l * v * v.transpose()); 
        eigenvalues.emplace_back(std::move(l));
        eigenvectors.emplace_back(std::move(v));
    }

    return std::make_tuple(eigenvalues, eigenvectors);
 }
