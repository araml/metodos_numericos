#include <cassert>
#include <iostream>
#include <fstream>
#include <math.h>
#include <power_iteration.h>
#include <set>
#include <config_tests.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::EigenSolver;

float EPSILON = pow(10, -20);

void test_check_eigen_values() { 
    MatrixXd B(2, 2);
    B << 1, 0, 0, -2;
    VectorXd v(2);
    v << 8, 3;
    
    auto [l1, v1, i1] = power_iteration_method(B, v, 5000, EPSILON);
    auto H = B - (l1 * v1 * v1.transpose());
    auto [l2, v2, i2] = power_iteration_method(H, v, 5000, EPSILON);
    
    // we don't know which eigenvalue we get first 
    std::set<float> s1{l1, l2};
    std::set<float> s2{1, -2};
    assert(s1 == s2);
    std::cout << "check_eigenvalues ok" << std::endl;
    std::cout << "\teigenvalue " << l1 << " found in " << i1 << " iterations" << std::endl;
    std::cout << "\teigenvalue " << l2 << " found in " << i2 << " iterations" << std::endl;
}

void test_deflate_single_eigen_value() { 
    MatrixXd B(2, 2);
    B << 1, 0, 0, -2;
    VectorXd v(2);
    v << 1, 1;

    auto [ls, vs] = deflate_impl(B, v, 5000, 1, EPSILON);
    
    std::set<float> s(ls.begin(), ls.end());
    // We need to have *only* one of these 
    // but we don't know the order we just count the occurrences of both 
    auto exists = s.count(1) + s.count(-2);
    assert(exists == 1);
    std::cout << "deflate_single_eigenvalue ok" << std::endl;
}

void test_deflate_all_eigen_values() { 
    MatrixXd B(2, 2);
    B << 1, 0, 0, -2;
    VectorXd v(2);
    v << 1, 1;

    auto [ls, vs] = deflate_impl(B, v, 5000, 2, EPSILON);
    
    std::set<float> s1(ls.begin(), ls.end());
    std::set<float> s2{1, -2};
    assert(s1 == s2);    
    std::cout << "deflate_all_eigenvalues ok" << std::endl;

    std::cout << "our eigenvalues" << std::endl;
    for (auto l : ls) 
        std::cout << l << std::endl;
    std::cout << "our eigenvectors" << std::endl;
    for (auto &v : vs) 
        std::cout << "(" << v.transpose() << ")" << std::endl;

    EigenSolver<MatrixXd> es(B);
    std::cout << "eigenvalues " << std::endl << es.eigenvalues() << std::endl << std::endl;
    std::cout << "eigenvectors " << std::endl << es.eigenvectors() << std::endl << std::endl; 
}

void test_incorrect_number_of_eigenvalues() { 
    MatrixXd B(2, 2);
    B << 1, 0, 0, -2;
    VectorXd v(2);
    v << 1, 1;

    try { 
    auto [ls, vs] = deflate_impl(B, v, 5000, 10, EPSILON);
    } catch (...) { 
        return;
    }

    assert(0);
}

int main() {
    test_check_eigen_values();
    test_deflate_single_eigen_value();
    test_deflate_all_eigen_values();
    test_incorrect_number_of_eigenvalues();
    return 0;
}
