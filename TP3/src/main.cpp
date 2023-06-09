#include <tuple>
#include <iostream>
#include <metodos_iterativos.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(iterative_methods, m) {
    m.doc() = "iterative_methods";
    m.def("gaussian_elimination", &gaussian_elimination);

    m.def("jacobi_matrix", &jacobi_matrix,
          py::arg(),  
          py::arg(),  
          py::arg(),  
          py::arg("iterations") = 10000, 
          py::arg("eps") = double(1e-6),
          py::arg("debug") = false);

    m.def("gauss_seidel_matrix", &gauss_seidel_matrix,
          py::arg(),  
          py::arg(),  
          py::arg(),  
          py::arg("iterations") = 10000, 
          py::arg("eps") = double(1e-6),
          py::arg("debug") = false);
    
    m.def("jacobi_sum_method", &jacobi_sum_method,
          py::arg(),  
          py::arg(),  
          py::arg(),  
          py::arg("iterations") = 10000, 
          py::arg("eps") = double(1e-6),
          py::arg("debug") = false);
    
    m.def("gauss_seidel_sum_method", &gauss_seidel_sum_method, 
          py::arg(),  
          py::arg(),  
          py::arg(),  
          py::arg("iterations") = 10000, 
          py::arg("eps") = double(1e-6),
          py::arg("debug") = false);
}
