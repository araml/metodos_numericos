#include <fstream>
#include <iostream>
#include <metodos_iterativos.h>
#include <stdexcept>
#include <string>
#include <ranges>

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
    MatrixXd D_inverse = D.inverse();
    MatrixXd L_plus_U = L + U;

    VectorXd x = VectorXd::Random(n);
    for (int i = 0; i < iterations; i++) {
        VectorXd prev_x = x;
        x = D_inverse * (L_plus_U*x + b);
        if ((x - prev_x).norm() < eps) {
            return x;
        }
    }

    throw std::logic_error("Matrix does not converge");
    return x;
}

VectorXd gauss_seidel_matrix(const MatrixXd &matrix,
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
    MatrixXd D_minus_L_inverse = (D - L).inverse();

    VectorXd x = VectorXd::Random(n);
    for (int i = 0; i < iterations; i++) {
        VectorXd prev_x = x;
        x = D_minus_L_inverse * (U*x + b);
        if ((x - prev_x).norm() < eps) {
            return x;
        }
    }

    throw std::logic_error("Matrix does not converge");
    return x;
}

// Esto es jacobi me parece..
VectorXd gaussSeidel(MatrixXd &matrix, VectorXd &b, int iterations, double eps) {
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

VectorXd jacobi_sum_method(const MatrixXd &m, VectorXd &b, 
                           int iterations, double eps) {
    int n = m.cols();
    VectorXd x = VectorXd::Random(n);
    for (int iter : std::views::iota(0, iterations)) {
        VectorXd x_ant = x;
        for (int i : std::views::iota(0, n)) { 
            float x_i = b(i);
            for (size_t j : std::views::iota(0, n)) { 
                if (i != j)
                    x_i -= m.coeff(i, j) * x_ant(j);
            }
            x_i *= 1/m.coeff(i, i);
            x(i) = x_i;
        }
        if ((x - x_ant).norm() < eps) { 
            return x;
        }
    }
    return x;
}

VectorXd gauss_seidel_sum_method(const MatrixXd &m, VectorXd &b,
                                 int iterations, double eps) { 
    int n = m.cols();
    VectorXd x = VectorXd::Random(n);
    for (int iter : std::views::iota(0, iterations)) {
        VectorXd x_ant = x;
        for (int i : std::views::iota(0, n)) { 
            float x_i = b(i);
            for (size_t j : std::views::iota(i + 1, n)) { 
                if (i != j)
                    x_i -= m.coeff(i, j) * x_ant(j);
            }
            
            for (size_t j : std::views::iota(0, i)) { 
                x_i -= m.coeff(i, j) * x(j);
            }

            x_i *= 1/m.coeff(i, i);
            x(i) = x_i;
        }
        if ((x - x_ant).norm() < eps) { 
            return x;
        }
    }
    return x;
}
