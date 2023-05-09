#include <Eigen/Dense>
#include <iostream>
#include <tuple>

std::tuple<double, VectorXd>
metodo_potencia(const MatrixXd &B, const VectorXd &x_0, size_t iters) {
    VectorXd v = x_0;
    for (size_t i = 0; i < iters; i++) {
        auto bv = B * v;
        v = bv / bv.norm();
    }
    double lambda = v.transpose() * B * v;

    return std::make_tuple(lambda, v);
}

int main() {
    MatrixXd B(2, 2);
    B << 1, 0, 0, -2;
    VectorXd v(2);
    v << 1, 1;
    std::cout << B << std::endl;
    auto [l, v1] = metodo_potencia(B, v, 5000);
    std::cout << "eigenvalue: " << l << std::endl
              << " eigenvector: " << v1 << std::endl;

    auto H = B - (l * v1 * v1.transpose());
    std::cout << "Householder?: " << std::endl << H << std::endl;
    std::cout << H * v1 << std::endl;
    auto [l2, v2] = metodo_potencia(H, v, 5000);
    std::cout << "eigenvalue: " << l2 << std::endl
              << " eigenvector: " << v2 << std::endl;
    return 0;
}
