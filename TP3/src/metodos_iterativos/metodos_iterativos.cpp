#include <fstream>
#include <iostream>
#include <metodos_iterativos.h>
#include <stdexcept>
#include <string>

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;


VectorXd ones(int size) {
    VectorXd res(size);
    for (int i = 0; i < size; i++) {
        res(i) = 1;
    }
    return res;
}

VectorXd jacobi_matrix(const MatrixXd &matrix,
                       const VectorXd &b,
                       int iterations,
                       double eps) {
    int n = matrix.cols();
    for (int i = 0; i < n; i++) {
        if (matrix(i, i) == 0)
            throw std::logic_error("Matrix has zeros in diagonal");
    }
    MatrixXd D = matrix.diagonal().asDiagonal();
    MatrixXd L = (-matrix).triangularView<Eigen::StrictlyLower>();
    MatrixXd U = (-matrix).triangularView<Eigen::StrictlyUpper>();
    MatrixXd DInverse = D.inverse();
    MatrixXd LplusU = L + U;

    VectorXd x = VectorXd::Random(n);
    for (int i = 0; i < iterations; i++) {
        VectorXd prev_x = x;
        x = DInverse * (LplusU*x + b);
        if ((x - prev_x).norm() < eps) {
            return x;
        }
    }

    throw std::logic_error("Matrix does not converge");
    return x;
}

VectorXd gaussSeidel(MatrixXd &matrix, VectorXd &b, int iterations=10000, double eps=1e-6) {
    int n = matrix.cols();
    VectorXd x = VectorXd::Random(n);
    for (int iter = 0; iter < iterations; iter++) {
            VectorXd x_ant = x;
            for (int i = 0; i < n; i++) {
                double sum = 0;
                for (int j = 0; j < n; j++) {
                    if (j == i) { continue; }
                    sum += matrix.coeff(i, j) * x(j);
                }
                x(i) = (b(i) - sum) / matrix.coeff(i, i);
            }
            if ((x - x_ant).norm() < eps) {
                return x / x.sum();
            }
    }
    return x / x.sum();
}

VectorXd gaussianElimination(MatrixXd& matrix, VectorXd &b) {
    Eigen::FullPivLU<Eigen::MatrixXd> lu(matrix);
    VectorXd x = lu.solve(b);
    return x;
}