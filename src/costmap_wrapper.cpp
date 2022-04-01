#include "../include/costmap_wrapper.h"

std::string CostmapWrapper::shareCostMap() {
    return "hello world!";
}

namespace py = pybind11;

PYBIND11_MODULE(mymodule, m) {
    py::class_<CostmapWrapper>(m, "mymodule")
        .def(py::init())
        .def("shareCostMap", &CostmapWrapper::shareCostMap);
}
