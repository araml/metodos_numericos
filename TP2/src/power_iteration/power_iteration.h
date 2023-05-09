#pragma once
#include <Eigen/Dense>

std::tuple<double, Eigen::VectorXd>
metodo_potencia(const Eigen::MatrixXd &B, const Eigen::VectorXd &x_0, 
                size_t iters, double eps = 0.0f);
