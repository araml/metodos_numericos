#pragma once
#include <Eigen/Dense>

std::tuple<float, Eigen::VectorXd, size_t>
power_iteration_method(const Eigen::MatrixXd &B, const Eigen::VectorXd &x_0, 
                size_t iters, float eps = 0.0f);

std::tuple<std::vector<float>, std::vector<Eigen::VectorXd>>
deflate_impl(Eigen::MatrixXd M, const Eigen::VectorXd &x_0, size_t iters, 
        size_t number_of_eigenvalues, float eps = 0.0f);

std::tuple<Eigen::MatrixXd, size_t, float>
read_matrix_iterations_tolerance(std::ifstream &infile);

std::tuple<std::vector<float>, std::vector<Eigen::VectorXd>>
deflate(std::string filename, const Eigen::VectorXd &x_0, 
        size_t number_of_eigenvalues);
