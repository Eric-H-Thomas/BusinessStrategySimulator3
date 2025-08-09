/*
This main function is used to run the simulator on its own, with control and/or pre-trained AI agents.
Use business_strategy_gym_env.py or another Python script for training AI agents.
*/

#include <exception>
#include <iostream>
#include <string>
#include <stdexcept>
#include "Simulator/Simulator.h"
#include "Config/ConfigValidator.h"
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/pybind11.h>

using std::cout;
using std::endl;

namespace py = pybind11;

int main(int argc, char* argv[]) {

    try {
        // Check for correct number of command-line arguments
        if (argc != 2) {
            throw std::invalid_argument("Expected 1 command-line argument. Got " + std::to_string(argc - 1));
        }

        // The scoped interpreter allows C++ to interpret Python as long as the interpreter is in scope
        py::scoped_interpreter guard{};

        // py::object is the C++ side’s generic reference to any Python value, such as a number, string, function,
        // list, or user-defined class instance
        py::object simulate_function;

        // Add the project root (which contains simulator.py) to the Python path
        py::module::import("sys").attr("path").attr("append")("../");

        // Import the script
        py::module script = py::module::import("simulator");

        // Set the simulate function py::object equal to the simulator.py simulate function
        simulate_function = script.attr("simulate");

        // Initialize the simulator, validate and load the configs, and prepare the simulator to run
        Simulator simulator;
        validate_config(argv[1]);
        simulator.load_json_configs(argv[1]);
        simulator.prepare_to_run();

        // Loop over the simulations
        for (int iSim = 0; iSim < simulator.get_num_sims(); iSim++) {
            // Reflect to the CRT the index of the current simulation
            cout << "Beginning simulation " << iSim << " of " << simulator.get_num_sims() - 1 << " (indexed at 0)" << endl;

            // Reset and run the simulator
            simulator.reset();
            simulator.run(simulate_function);
        }

        // Generate the output files
        if (simulator.bGenerateMasterOutput) {
            simulator.masterHistory.generate_master_output();
            simulator.masterHistory.generate_market_overlap_file();
        }
    }

    catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }

    return 0;
}