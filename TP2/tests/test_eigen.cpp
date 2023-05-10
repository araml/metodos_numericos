#include <set>
#include <cassert>
#include <iostream>
#include <power_iteration.h>

void test_check_eigen_values() { 
    Eigen::MatrixXd B(2, 2);
    B << 1, 0, 0, -2;
    Eigen::VectorXd v(2);
    v << 1, 1;
    
    auto [l1, v1] = power_iteration_method(B, v, 5000);
    auto H = B - (l1 * v1 * v1.transpose());
    auto [l2, v2] = power_iteration_method(H, v, 5000);
    
    // we don't know which eigenvalue we get first 
    std::set<float> s1{l1, l2};
    std::set<float> s2{1, -2};
    assert(s1 == s2);
}

void test_deflate_single_eigen_value() { 
    Eigen::MatrixXd B(2, 2);
    B << 1, 0, 0, -2;
    Eigen::VectorXd v(2);
    v << 1, 1;

    auto [ls, vs] = deflate(B, v, 5000, 1);
    
    std::set<float> s(ls.begin(), ls.end());
    // We need to have *only* one of these 
    // but we don't know the order we just count the occurrences of both 
    auto exists = s.count(1) + s.count(-2);
    assert(exists == 1);
}

void test_deflate_all_eigen_values() { 
    Eigen::MatrixXd B(2, 2);
    B << 1, 0, 0, -2;
    Eigen::VectorXd v(2);
    v << 1, 1;

    auto [ls, vs] = deflate(B, v, 5000, 2);
    
    std::set<float> s1(ls.begin(), ls.end());
    std::set<float> s2{1, -2};
    assert(s1 == s2);    
}

void test_incorrect_number_of_eigenvalues() { 
    Eigen::MatrixXd B(2, 2);
    B << 1, 0, 0, -2;
    Eigen::VectorXd v(2);
    v << 1, 1;

    try { 
    auto [ls, vs] = deflate(B, v, 5000, 10);
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
