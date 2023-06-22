#include <cassert>
#include <fstream>
#include <iostream>
#include <math.h>
#include <metodos_iterativos.h>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::EigenSolver;

float EPSILON = 1e-6;

TEST_CASE("test matrix with zeros in the diagonal") {
    MatrixXd M(2, 2);
    M << 0, 1, 1, 1;
    VectorXd b = ones(2);
    VectorXd x_0 = ones(2);
    
    REQUIRE_THROWS_WITH(jacobi_matrix(M, b, x_0), 
                        doctest::Contains("Matrix has zeros in diagonal"));
    REQUIRE_THROWS_WITH(jacobi_sum_method(M, b, x_0), 
                        doctest::Contains("Matrix has zeros in diagonal"));
}

TEST_CASE("test non convergent matrix") {
    MatrixXd M(2, 2);
    M << 0.5, 1, 1, 0.5;
    VectorXd b = ones(2);
    VectorXd x_0 = ones(2);
    
    REQUIRE_THROWS_WITH(jacobi_matrix(M, b, x_0),
                        doctest::Contains("Matrix does not converge"));
    REQUIRE_THROWS_WITH(jacobi_sum_method(M, b, x_0),
                        doctest::Contains("Matrix does not converge"));
}

TEST_CASE("test convergent matrix") {
    MatrixXd M(2, 2);
    M << 2, 1, 1, 2;
    VectorXd b = ones(2);
    VectorXd x_0 = ones(2);
    
    Eigen::FullPivLU<Eigen::MatrixXd> lu(M);
    auto expected = gaussian_elimination(M, b);
    auto [j, i] = jacobi_matrix(M, b, x_0, 1000, EPSILON);

    CHECK((expected-j).norm() < EPSILON);
}

TEST_CASE("test matrix vs sum method") { 
    MatrixXd M(2, 2);
    M << 2, 1, 1, 2;
    VectorXd b = ones(2);
    VectorXd x_0 = ones(2);
    
    Eigen::FullPivLU<Eigen::MatrixXd> lu(M);
    auto [x1, i1] = jacobi_matrix(M, b, x_0, 1000, EPSILON);
    auto [x2, i2] = jacobi_sum_method(M, b, x_0, 1000, EPSILON);

    CHECK((x1 - x2).norm() < EPSILON);
}


