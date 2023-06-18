#include <iostream>
#include "eigen/Eigen/Dense"
#include <fstream>
#include <string>
#include <stdexcept>

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;


VectorXd ones(int size) {
  VectorXd res(size);
  for (int i = 0; i < size; i++) {
    res(i) = 1;
  }
  return res;
}

VectorXd jacobi(MatrixXd &matrix, int niter=10000, double eps=1e-6){
  VectorXd b = ones(matrix.rows());
  MatrixXd D = matrix.diagonal().asDiagonal();
  MatrixXd L = (-matrix).triangularView<Eigen::StrictlyLower>();
  MatrixXd U = (-matrix).triangularView<Eigen::StrictlyUpper>();
  MatrixXd DMinus1= D.inverse();
  MatrixXd LplusU = L + U;

  VectorXd x = VectorXd::Random(matrix.cols());
  for(int i = 0; i< niter; i++){
    VectorXd x_ant = x;
    x = DMinus1 * ((LplusU) * (x)) + (DMinus1 * b);
    if ((x - x_ant).norm() < eps) {
      return x / x.sum();
    }
  }

  return x / x.sum();
}

VectorXd gaussSeidel(MatrixXd &matrix, int niter=10000, double eps=1e-6) {
  int n = matrix.cols();
  VectorXd b = ones(matrix.rows());
  VectorXd x = VectorXd::Random(n);
  for (int iter = 0; iter < niter; iter++) {
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

VectorXd gaussianElimination(MatrixXd& matrix) {
    VectorXd b = ones(matrix.rows());
    Eigen::FullPivLU<Eigen::MatrixXd> lu(matrix);
    VectorXd x = lu.solve(b);
    return x / x.sum();
}



