#include <fstream>
#include <iostream>
#include <metodos_iterativos.h>
#include <ranges>
#include <stdexcept>
#include <string>

using Eigen::MatrixXd;
using Eigen::VectorXd;


VectorXd ones(int size) {
    VectorXd res(size);
    for (int i = 0; i < size; i++) {
        res(i) = 1;
    }
    return res;
}

VectorXd gaussian_elimination(MatrixXd& m, VectorXd &b) {
    Eigen::FullPivLU<Eigen::MatrixXd> lu(m);
    VectorXd x = lu.solve(b);
    return x;
}

std::tuple<VectorXd, int> jacobi_matrix(
        const MatrixXd &m,
        const VectorXd &b,
        const VectorXd &x_0,
        int iterations,
        double eps) {
    int n = m.cols();
    for (int i = 0; i < n; i++) {
        if (m(i, i) == 0)
            throw std::logic_error("Matrix has zeros in diagonal");
    }
    MatrixXd D = m.diagonal().asDiagonal();
    MatrixXd L = (-m).triangularView<Eigen::StrictlyLower>();
    MatrixXd U = (-m).triangularView<Eigen::StrictlyUpper>();
    MatrixXd D_inverse = D.inverse();
    MatrixXd L_plus_U = L + U;

    VectorXd x = x_0;
    for (int i = 0; i < iterations; i++) {
        VectorXd prev_x = x;
        x = D_inverse * (L_plus_U*x + b);
        if ((x - prev_x).norm() < eps) {
            return std::make_tuple(x, i);
        }
    }

    throw std::logic_error("Matrix does not converge");
    return std::make_tuple(x, -1);
}

std::tuple<VectorXd, int> gauss_seidel_matrix(
        const MatrixXd &m,
        const VectorXd &b,
        const VectorXd &x_0,
        int iterations,
        double eps) {
    int n = m.cols();
    for (int i = 0; i < n; i++) {
        if (m(i, i) == 0)
            throw std::logic_error("Matrix has zeros in diagonal");
    }
    MatrixXd D = m.diagonal().asDiagonal();
    MatrixXd L = (-m).triangularView<Eigen::StrictlyLower>();
    MatrixXd U = (-m).triangularView<Eigen::StrictlyUpper>();
    MatrixXd D_minus_L_inverse = (D - L).inverse();

    VectorXd x = x_0;   
    for (int i = 0; i < iterations; i++) {
        VectorXd prev_x = x;
        x = D_minus_L_inverse * (U*x + b);
        if ((x - prev_x).norm() < eps) {
            return std::make_tuple(x, i);
        }
    }

    throw std::logic_error("Matrix does not converge");
    return std::make_tuple(x, -1);
}

std::tuple<VectorXd, int> jacobi_sum_method(
        const MatrixXd &m,
        const VectorXd &b, 
        const VectorXd &x_0,
        int iterations,
        double eps) {
    int n = m.cols();
    VectorXd x = x_0;
    for (int iter : std::views::iota(0, iterations)) {
        VectorXd prev_x = x;
        for (int i : std::views::iota(0, n)) {
            if (m.coeff(i, i) == 0)
                throw std::logic_error("Matrix has zeros in diagonal");
            float x_i = b(i);
            for (size_t j : std::views::iota(0, n)) { 
                if (i != j)
                    x_i -= m.coeff(i, j) * prev_x(j);
            }
            x_i *= 1/m.coeff(i, i);
            x(i) = x_i;
        }
        if ((x - prev_x).norm() < eps) { 
            return std::make_tuple(x, iter);
        }
    }
    throw std::logic_error("Matrix does not converge");
    return std::make_tuple(x, -1);
}

std::tuple<VectorXd, int> gauss_seidel_sum_method(
        const MatrixXd &m,
        const VectorXd &b,
        const VectorXd &x_0,
        int iterations,
        double eps) {
    int n = m.cols();
    VectorXd x = x_0;
    for (int iter : std::views::iota(0, iterations)) {
        VectorXd prev_x = x;
        for (int i : std::views::iota(0, n)) {
            if (m.coeff(i, i) == 0)
                throw std::logic_error("Matrix has zeros in diagonal");
            float x_i = b(i);
            for (size_t j : std::views::iota(i + 1, n)) { 
                if (i != j)
                    x_i -= m.coeff(i, j) * prev_x(j);
            }
            
            for (size_t j : std::views::iota(0, i)) { 
                x_i -= m.coeff(i, j) * x(j);
            }

            x_i *= 1/m.coeff(i, i);
            x(i) = x_i;
        }
        if ((x - prev_x).norm() < eps) { 
            return std::make_tuple(x, iter);
        }
    }
    throw std::logic_error("Matrix does not converge");
    return std::make_tuple(x, -1);
}
