//
// Created by Eric Thomas on 12/2/23.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../WorkingFiles/PythonAPI/PythonAPI.h"

PYBIND11_MODULE(simulator_module, m) {
    // Expose the API class
    pybind11::class_<PythonAPI>(m, "PythonAPI")
            .def(pybind11::init<>())
            .def("init_simulator", &PythonAPI::init_simulator)
            .def("step", &PythonAPI::step)
            .def("reset", &PythonAPI::reset)
            .def("close", &PythonAPI::close)
            .def("get_num_markets", &PythonAPI::get_num_markets)
            .def("get_num_agents", &PythonAPI::get_num_agents);
}