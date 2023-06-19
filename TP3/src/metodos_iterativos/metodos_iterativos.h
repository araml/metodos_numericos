#pragma once
#include <Eigen/Dense>

Eigen::VectorXd ones(int size);

Eigen::VectorXd jacobi_matrix(const Eigen::MatrixXd &matrix,
                              const Eigen::VectorXd &b,
                              int iterations=10000,
                              double eps=1e-6);

Eigen::VectorXd gaussianElimination(Eigen::MatrixXd& matrix,
                                    Eigen::VectorXd &b);