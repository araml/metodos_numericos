#include <power_iteration.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;

std::tuple<double, VectorXd>
metodo_potencia(const MatrixXd &B, const VectorXd &x_0, size_t iters,
                double eps) {
    VectorXd v = x_0;
    for (size_t i = 0; i < iters; i++) {
        auto bv = B * v;
        v = bv / bv.norm();
    }
    double lambda = v.transpose() * B * v;

    return std::make_tuple(lambda, v);
}
