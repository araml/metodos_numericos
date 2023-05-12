#pragma once
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;

std::tuple<float, VectorXd, size_t>
power_iteration_method(const MatrixXd &B, const VectorXd &x_0, 
                size_t iters, float eps = 0.0f);

std::tuple<std::vector<float>, std::vector<VectorXd>>
deflate_impl(MatrixXd M, const VectorXd &x_0, size_t iters, 
        size_t number_of_eigenvalues, float eps = 0.0f);

std::tuple<MatrixXd, size_t, float>
read_matrix_iterations_tolerance(std::ifstream &infile);

std::tuple<std::vector<float>, std::vector<VectorXd>>
deflate(std::string filename, const VectorXd &x_0, size_t number_of_eigenvalues);