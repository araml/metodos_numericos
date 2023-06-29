#pragma once
#include <Eigen/Dense>

Eigen::VectorXd ones(int size);

Eigen::VectorXd gaussian_elimination(Eigen::MatrixXd& matrix,
                                     Eigen::VectorXd &b);

std::tuple<Eigen::VectorXd, int> jacobi_matrix(
    const Eigen::MatrixXd &matrix,
    const Eigen::VectorXd &b,
    const Eigen::VectorXd &x_0,
    int iterations = 10000,
    double eps = 1e-6,
    bool debug = false);

std::tuple<Eigen::VectorXd, int> gauss_seidel_matrix(
    const Eigen::MatrixXd &matrix,
    const Eigen::VectorXd &b,
    const Eigen::VectorXd &x_0,
    int iterations = 10000,
    double eps = 1e-6,
    bool debug = false);

std::tuple<Eigen::VectorXd, int> jacobi_sum_method(
    const Eigen::MatrixXd &matrix,
    const Eigen::VectorXd &b,
    const Eigen::VectorXd &x_0,
    int iterations = 10000,
    double eps = 1e-6,
    bool debug = false);

std::tuple<Eigen::VectorXd, int> gauss_seidel_sum_method(
    const Eigen::MatrixXd &matrix, 
    const Eigen::VectorXd &b,
    const Eigen::VectorXd &x_0,
    int iterations = 10000,
    double eps = 1e-6, 
    bool debug = false);
