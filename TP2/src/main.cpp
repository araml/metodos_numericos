#include <tuple>
#include <iostream>
#include <power_iteration.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(deflate, m){
    m.doc() = "powre_iteration";
    m.def("power_iteration", &power_iteration_method);
}
