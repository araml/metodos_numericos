#pragma once
#include <Eigen/Dense>

Eigen::VectorXd ones(int size);

Eigen::VectorXd jacobi_matrix(const Eigen::MatrixXd &matrix,
                              const Eigen::VectorXd &b,
                              int iterations=10000,
                              double eps=1e-6);

Eigen::VectorXd gaussianElimination(Eigen::MatrixXd& matrix,
                                    Eigen::VectorXd &b);

Eigen::VectorXd gauss_seidel_matrix(const Eigen::MatrixXd &matrix,
                                    const Eigen::VectorXd &b,
                                    int iterations=10000,
                                    double eps=1e-6);


Eigen::VectorXd jacobi_sum_method(const Eigen::MatrixXd &matrix, 
                                  Eigen::VectorXd &b, 
                                  int iterations = 10000,
                                  double eps = 1e-6);

Eigen::VectorXd gauss_seidel_sum_method(const Eigen::MatrixXd &matrix, 
                                        Eigen::VectorXd &b, 
                                        int iterations = 10000,
                                        double eps = 1e-6);
