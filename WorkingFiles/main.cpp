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

using std::cout;
using std::endl;


int main(int argc, char* argv[]) {

    try {
        // Check for correct number of command-line arguments
        if (argc != 2) {
            throw std::invalid_argument("Expected 1 command-line argument. Got " + std::to_string(argc - 1));
        }

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
            simulator.run();
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