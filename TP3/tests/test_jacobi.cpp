#include <cassert>
#include <fstream>
#include <iostream>
#include <math.h>
#include <metodos_iterativos.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::EigenSolver;

float EPSILON = 1e-6;

void test_diagonal_zeros_matrix() {
    MatrixXd M(2, 2);
    M << 0, 1, 1, 1;
    VectorXd b = ones(2);

    try {
        auto j = jacobi_matrix(M, b);
    } catch (std::logic_error const& ex) {
        const char* message = ex.what();
        const char* expected = "Matrix has zeros in diagonal";
        assert(strcmp(message, expected) == 0);
        std::cout << "Matrix with zeros in diagonal OK" << std::endl;
        return;
    }

    assert(0);
}

void test_non_convergent_matrix() {
    MatrixXd M(2, 2);
    M << 0.5, 1, 1, 0.5;
    VectorXd b = ones(2);
    
    try {
        auto j = jacobi_matrix(M, b);
    } catch (std::logic_error const& ex) {
        const char* message = ex.what();
        const char* expected = "Matrix does not converge";
        assert(strcmp(message, expected) == 0);
        std::cout << "Non-convergent matrix OK" << std::endl;
        return;
    }

    assert(0);
}

void test_convergent_matrix() {
    MatrixXd M(2, 2);
    M << 2, 1, 1, 2;
    VectorXd b = ones(2);
    
    Eigen::FullPivLU<Eigen::MatrixXd> lu(M);
    auto expected = gaussianElimination(M, b);
    auto j = jacobi_matrix(M, b, 1000, EPSILON);

    assert((expected-j).norm() < EPSILON);
    std::cout << "Convergent matrix OK" << std::endl;
}

void test_matrix_vs_sum_method() { 
    MatrixXd M(2, 2);
    M << 2, 1, 1, 2;
    VectorXd b = ones(2);
    
    Eigen::FullPivLU<Eigen::MatrixXd> lu(M);
    auto x1 = jacobi_matrix(M, b, 1000, EPSILON);
    auto x2 = jacobi_sum_method(M, b, 1000, EPSILON);

    assert((x1 - x2).norm() < EPSILON);
    std::cout << "Test sum vs matrix method" << std::endl;
}

int main() {
    test_diagonal_zeros_matrix();
    test_non_convergent_matrix();
    test_convergent_matrix();
}
