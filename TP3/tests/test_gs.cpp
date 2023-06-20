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

    REQUIRE_THROWS_WITH(gauss_seidel_matrix(M, b),
                        doctest::Contains("Matrix has zeros in diagonal"));
    REQUIRE_THROWS_WITH(gauss_seidel_sum_method(M, b),
                        doctest::Contains("Matrix has zeros in diagonal"));
}

TEST_CASE("test non convergent matrix") {
    MatrixXd M(2, 2);
    M << 0.5, 1, 1, 0.5;
    VectorXd b = ones(2);
    
    REQUIRE_THROWS_WITH(gauss_seidel_matrix(M, b),
                        doctest::Contains("Matrix does not converge"));
    REQUIRE_THROWS_WITH(gauss_seidel_sum_method(M, b),
                        doctest::Contains("Matrix does not converge"));
}

TEST_CASE("test convergent matrix") {
    MatrixXd M(2, 2);
    M << 2, 1, 1, 2;
    VectorXd b = ones(2);
    
    Eigen::FullPivLU<Eigen::MatrixXd> lu(M);
    auto expected = gaussianElimination(M, b);
    auto j = gauss_seidel_matrix(M, b, 1000, EPSILON);

    CHECK((expected-j).norm() < EPSILON);
}

TEST_CASE("test compare gauss seidel matrix vs sum method") {
    MatrixXd M(2, 2);
    M << 2, 1, 1, 2;
    VectorXd b = ones(2);
    
    Eigen::FullPivLU<Eigen::MatrixXd> lu(M);
    auto v1 = gauss_seidel_matrix(M, b, 1000, EPSILON);
    auto v2 = gauss_seidel_sum_method(M, b, 1000, EPSILON);

    CHECK((v1 - v2).norm() < EPSILON);
}
