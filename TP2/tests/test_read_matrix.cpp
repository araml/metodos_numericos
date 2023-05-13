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

void test_read_matrix_of_ints() {
    MatrixXd expected_matrix(2, 2);
    expected_matrix << 1, 0, 0, 1;
    std::string filename = std::string(file_folder) + "/matrix_of_ints.txt";
    std::ifstream infile(filename);

    auto[actual_matrix, actual_iterations, actual_tolerance] = 
        read_matrix_iterations_tolerance(infile);

    assert(actual_matrix.isApprox(expected_matrix));
    assert(actual_iterations == 10);
    assert(actual_tolerance == 0.000001f); 
    
    std::cout << "read_matrix_of_ints ok" << std::endl;
}

void test_read_matrix_of_floats() { 
    MatrixXd expected_matrix(2, 2);
    expected_matrix << 0.2, 0, 0, 1;
    size_t expected_iterations = 10;
    float expected_tolerance = 0.000001;
    std::string filename = std::string(file_folder) + "/matrix_of_floats.txt";
    std::ifstream infile(filename);
    
    auto[actual_matrix, actual_iterations, actual_tolerance] = 
        read_matrix_iterations_tolerance(infile);

    
    assert(actual_matrix.isApprox(expected_matrix));

    std::cout << "read_matrix_of_floats ok" << std::endl;
}

int main() {
    test_read_matrix_of_ints();
    test_read_matrix_of_floats();
    return 0;
}
