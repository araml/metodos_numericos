#include <set>
#include <cassert>
#include <iostream>
#include <power_iteration.h>

void test_check_eigen_values() { 
    Eigen::MatrixXd B(2, 2);
    B << 1, 0, 0, -2;
    Eigen::VectorXd v(2);
    v << 1, 1;
    
    auto [l1, v1] = metodo_potencia(B, v, 5000);
    auto H = B - (l1 * v1 * v1.transpose());
    auto [l2, v2] = metodo_potencia(H, v, 5000);
    
    // we don't know which eigenvalue we get first 
    std::set<double> s1{l1, l2};
    std::set<double> s2{1, -2};
    assert(s1 == s2);
}


int main() {
    test_check_eigen_values();
    return 0;
}
